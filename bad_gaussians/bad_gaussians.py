"""
BAD-Gaussians model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Type, Union

import torch

from gsplat.project_gaussians import project_gaussians
from gsplat.rasterize import rasterize_gaussians
from gsplat.sh import spherical_harmonics

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig
from nerfstudio.model_components import renderers

from bad_gaussians.bad_camera_optimizer import (
    BadCameraOptimizer,
    BadCameraOptimizerConfig,
    TrajSamplingMode,
)
from bad_gaussians.bad_losses import EdgeAwareVariationLoss
from bad_gaussians.network import SpkRecon_Net,Multi_SpkRecon_Net


@dataclass
class BadGaussiansModelConfig(SplatfactoModelConfig):
    """BAD-Gaussians Model config"""

    _target: Type = field(default_factory=lambda: BadGaussiansModel)
    """The target class to be instantiated."""

    rasterize_mode: Literal["classic", "antialiased"] = "antialiased"
    """
    Classic mode of rendering will use the EWA volume splatting with a [0.3, 0.3] screen space blurring kernel. This
    approach is however not suitable to render tiny gaussians at higher or lower resolution than the captured, which
    results "aliasing-like" artifacts. The antialiased mode overcomes this limitation by calculating compensation factors
    and apply them to the opacities of gaussians to preserve the total integrated density of splats.

    However, PLY exported with antialiased rasterize mode is not compatible with classic mode. Thus many web viewers that
    were implemented for classic mode can not render antialiased mode PLY properly without modifications.
    Refs:
    1. https://github.com/nerfstudio-project/gsplat/pull/117
    2. https://github.com/nerfstudio-project/nerfstudio/pull/2888
    3. Yu, Zehao, et al. "Mip-Splatting: Alias-free 3D Gaussian Splatting." arXiv preprint arXiv:2311.16493 (2023).
    """

    camera_optimizer: BadCameraOptimizerConfig = field(default_factory=BadCameraOptimizerConfig)
    """Config of the camera optimizer to use"""

    cull_alpha_thresh: float = 0.005
    """Threshold for alpha to cull gaussians. Default: 0.1 in splatfacto, 0.005 in splatfacto-big."""

    densify_grad_thresh: float = 4e-4
    """[IMPORTANT] Threshold for gradient to densify gaussians. Default: 4e-4. Tune it smaller with complex scenes."""

    continue_cull_post_densification: bool = False
    """Whether to continue culling after densification. Default: True in splatfacto, False in splatfacto-big."""

    resolution_schedule: int = 250
    """training starts at 1/d resolution, every n steps this is doubled.
    Default: 250. Use 3000 with high resolution images (e.g. higher than 1920x1080).
    """

    num_downscales: int = 0
    """at the beginning, resolution is 1/2^d, where d is this number. Default: 0. Use 2 with high resolution images."""

    enable_absgrad: bool = False
    """Whether to enable absgrad for gaussians. (It affects param tuning of densify_grad_thresh)
    Default: False. Ref: (https://github.com/nerfstudio-project/nerfstudio/pull/3113)
    """

    tv_loss_lambda: Optional[float] = None
    """weight of total variation loss"""

    use_3dgs: bool = True
    """use 3DGS reblur optimization loss"""
    
    use_spike: bool = True
    """use spike-net optimization loss"""

    use_flip: bool = False
    """use flip operation to align pose and result"""
    
    use_multi_net: bool = False
    """use multi-input spike-net"""
    
    use_multi_reblur: bool = False
    """use multi-reblur loss function"""
    
    weight_3dgs: float = 1
    """3dgs loss weight"""

    weight_spike: float = 1
    """spike loss weight"""

    weight_joint: float = 1
    """mutual loss weight"""

class BadGaussiansModel(SplatfactoModel):
    """BAD-Gaussians Model

    Args:
        config: configuration to instantiate model
    """

    config: BadGaussiansModelConfig
    camera_optimizer: BadCameraOptimizer

    def __init__(self, config: BadGaussiansModelConfig, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        # Scale densify_grad_thresh by the number of virtual views
        self.config.densify_grad_thresh /= self.config.camera_optimizer.num_virtual_views
        # (Experimental) Total variation loss
        self.tv_loss = EdgeAwareVariationLoss(in1_nc=3)
        # recon-net config
        self.spike_in_length = 41
        self.voxel_in_length = 34
        self.use_spike = self.config.use_spike
        self.use_3dgs = self.config.use_3dgs
        self.use_flip = self.config.use_flip
        self.use_multi_net = self.config.use_multi_net
        self.use_multi_reblur = self.config.use_multi_reblur
        
        if self.use_multi_net == False:
            self.spike_net = SpkRecon_Net(input_dim=self.spike_in_length)
        else:
            self.spike_net = Multi_SpkRecon_Net(input_dim=self.spike_in_length,voxel_dim = self.voxel_in_length)
        self.weight_3dgs = self.config.weight_3dgs
        self.weight_spike = self.config.weight_spike
        self.weight_joint = self.config.weight_joint
        
         
    def populate_modules(self) -> None:
        super().populate_modules()
        self.camera_optimizer: BadCameraOptimizer = self.config.camera_optimizer.setup(
            num_cameras=self.num_train_data, device="cpu"
        )

    def forward(
            self,
            camera: Cameras,
            mode: TrajSamplingMode = "uniform",
            spike: torch.Tensor = torch.zeros(0)
    ) -> Dict[str, Union[torch.Tensor, List]]:
        return self.get_outputs(camera,mode = mode,spike = spike)

    def save_spike_net(self,path):
        torch.save(self.spike_net.state_dict(),path)

    def get_outputs(
            self, camera: Cameras,
            mode: TrajSamplingMode = "uniform",
            spike: torch.Tensor = torch.zeros(0)
    ) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a Camera and returns a dictionary of outputs.

        Args:
            camera: Input camera. This camera should have all the needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        if not isinstance(camera, Cameras):
            print("Called get_outputs with not a camera")
            return {}
        assert camera.shape[0] == 1, "Only one camera at a time"

        is_training = self.training and torch.is_grad_enabled()

        # BAD-Gaussians: get virtual cameras
        virtual_cameras = self.camera_optimizer.apply_to_camera(camera, mode)
        self.num_cam = len(virtual_cameras)

        if is_training:
            if self.config.background_color == "random":
                background = torch.rand(3, device=self.device)
            elif self.config.background_color == "white":
                background = torch.ones(3, device=self.device)
            elif self.config.background_color == "black":
                background = torch.zeros(3, device=self.device)
            else:
                background = self.background_color.to(self.device)
        else:
            # logic for setting the background of the scene
            if renderers.BACKGROUND_COLOR_OVERRIDE is not None:
                background = renderers.BACKGROUND_COLOR_OVERRIDE.to(self.device)
            else:
                background = self.background_color.to(self.device)
        if self.crop_box is not None and not is_training:
            crop_ids = self.crop_box.within(self.means).squeeze()
            if crop_ids.sum() == 0:
                rgb = background.repeat(int(camera.height.item()), int(camera.width.item()), 1)
                depth = background.new_ones(*rgb.shape[:2], 1) * 10
                accumulation = background.new_zeros(*rgb.shape[:2], 1)
                return {"rgb": rgb, "depth": depth, "accumulation": accumulation, "background": background}
        else:
            crop_ids = None

        camera_downscale = self._get_downscale_factor()

        for cam in virtual_cameras:
            cam.rescale_output_resolution(1 / camera_downscale)

        # BAD-Gaussians: render virtual views
        virtual_views_img = []
        virtual_views_alpha = []
        virtual_views_recon = []
        virtual_views_tfp = []
        idx = 0
        # voxel pre-calculation
        if len(spike.shape) == 3:
            spike_voxel = spike[20:-20]
            spike_voxel = torch.cat((spike_voxel[:spike_voxel.shape[0] // 2], spike_voxel[spike_voxel.shape[0] // 2 + 1:]), dim=0)
            spike_voxel = torch.sum(spike_voxel[None].reshape(-1,4,self.voxel_in_length,spike.shape[-2],spike.shape[-1]),dim = 1)
        for cam in virtual_cameras:
            # todo ----------------- Part 1 Spike-Recon -----------------
            if len(spike.shape) == 3:
                # spike input from 40 to 136 [40,136]
                if mode == "start":
                    spike_idx = 40 
                elif mode == "mid":
                    spike_idx = 88
                elif mode == "end":
                    spike_idx = 136
                elif mode == "uniform":
                    spike_idx = 40 + idx * (96 // (len(virtual_cameras) - 1)) # len(virtual_cameras): 13 or 7
                spike_roi = spike[spike_idx-self.spike_in_length//2:spike_idx+self.spike_in_length//2+1][None]
                if self.use_multi_net == False:
                    spike_recon = self.spike_net(spike_roi)[0,0][...,None].repeat(1,1,3)
                else:
                    index = (spike_idx - 40) / 96 
                    spike_recon = self.spike_net(spike_voxel,spike_roi,index)[0,0][...,None].repeat(1,1,3)
                spike_tfp = torch.mean(spike[spike_idx-20:spike_idx+21+1],dim = 0,keepdim=False)[...,None].repeat(1,1,3)
                virtual_views_recon.append(spike_recon)
                virtual_views_tfp.append(spike_tfp)
                idx += 1
            else:
                virtual_views_recon.append(spike)
                virtual_views_tfp.append(spike)
            # todo ----------------- Part 2 3DGS-Rendering -----------------
            # shift the camera to center of scene looking at center
            R = cam.camera_to_worlds[0, :3, :3]  # 3 x 3
            T = cam.camera_to_worlds[0, :3, 3:4]  # 3 x 1
            # flip the z axis to align with gsplat conventions
            R_edit = torch.diag(torch.tensor([1, -1, -1], device=self.device, dtype=R.dtype))
            R = R @ R_edit
            # analytic matrix inverse to get world2camera matrix
            R_inv = R.T
            T_inv = -R_inv @ T
            viewmat = torch.eye(4, device=R.device, dtype=R.dtype)
            viewmat[:3, :3] = R_inv
            viewmat[:3, 3:4] = T_inv
            # update last_size
            W, H = int(cam.width.item()), int(cam.height.item())
            self.last_size = (H, W)

            if crop_ids is not None:
                opacities_crop = self.opacities[crop_ids]
                means_crop = self.means[crop_ids]
                features_dc_crop = self.features_dc[crop_ids]
                features_rest_crop = self.features_rest[crop_ids]
                scales_crop = self.scales[crop_ids]
                quats_crop = self.quats[crop_ids]
            else:
                opacities_crop = self.opacities
                means_crop = self.means
                features_dc_crop = self.features_dc
                features_rest_crop = self.features_rest
                scales_crop = self.scales
                quats_crop = self.quats

            colors_crop = torch.cat((features_dc_crop[:, None, :], features_rest_crop), dim=1)
            BLOCK_WIDTH = 16  # this controls the tile size of rasterization, 16 is a good default
            self.xys, depths, self.radii, conics, comp, num_tiles_hit, cov3d = project_gaussians(
                means_crop,
                torch.exp(scales_crop),
                1,
                quats_crop / quats_crop.norm(dim=-1, keepdim=True),
                viewmat.squeeze()[:3, :],
                cam.fx.item(),
                cam.fy.item(),
                cam.cx.item(),
                cam.cy.item(),
                H,
                W,
                BLOCK_WIDTH,
            )  # type: ignore

            # rescale the camera back to original dimensions before returning
            cam.rescale_output_resolution(camera_downscale)

            if (self.radii).sum() == 0:
                rgb = background.repeat(H, W, 1)
                depth = background.new_ones(*rgb.shape[:2], 1) * 10
                accumulation = background.new_zeros(*rgb.shape[:2], 1)

                return {"rgb": rgb, "depth": depth, "accumulation": accumulation, "background": background}

            # Important to allow xys grads to populate properly
            if is_training:
                self.xys.retain_grad()

            if self.config.sh_degree > 0:
                viewdirs = means_crop.detach() - cam.camera_to_worlds.detach()[..., :3, 3]  # (N, 3)
                viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
                n = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
                rgbs = spherical_harmonics(n, viewdirs, colors_crop)
                rgbs = torch.clamp(rgbs + 0.5, min=0.0) # type: ignore
            else:
                rgbs = torch.sigmoid(colors_crop[:, 0, :])

            # rescale the camera back to original dimensions
            # cam.rescale_output_resolution(camera_downscale)
            assert (num_tiles_hit > 0).any()  # type: ignore

            # apply the compensation of screen space blurring to gaussians
            if self.config.rasterize_mode == "antialiased":
                alphas = torch.sigmoid(opacities_crop) * comp[:, None]
            elif self.config.rasterize_mode == "classic":
                alphas = torch.sigmoid(opacities_crop)
            rgb, alpha = rasterize_gaussians(  # type: ignore
                self.xys,
                depths,
                self.radii,
                conics,
                num_tiles_hit,  # type: ignore
                rgbs,
                alphas,
                H,
                W,
                BLOCK_WIDTH,
                background=background,
                return_alpha=True,
            )  # type: ignore
            alpha = alpha[..., None]
            rgb = torch.clamp(rgb, max=1.0)  # type: ignore
            virtual_views_img.append(rgb)
            virtual_views_alpha.append(alpha)
        depth_im = None
        # spike
        virtual_views_recon = torch.stack(virtual_views_recon,dim = 0)
        spike_blur = virtual_views_recon.mean(dim = 0)

        # spike-tfp
        virtual_views_tfp = torch.stack(virtual_views_tfp,dim = 0)
        
        # image
        virtual_views_img = torch.stack(virtual_views_img, dim=0)
        rgb = virtual_views_img.mean(dim=0)
        # alpha
        virtual_views_alpha = torch.stack(virtual_views_alpha, dim=0)
        alpha = virtual_views_alpha.mean(dim=0)

        # eval
        if not is_training:
            depth_im = rasterize_gaussians(  # type: ignore
                self.xys,
                depths,
                self.radii,
                conics,
                num_tiles_hit,  # type: ignore
                depths[:, None].repeat(1, 3),
                torch.sigmoid(opacities_crop),
                H,
                W,
                BLOCK_WIDTH,
                background=torch.zeros(3, device=self.device),
            )[..., 0:1]  # type: ignore
            depth_im = torch.where(alpha > 0, depth_im / alpha, depth_im.detach().max())
        # ! important: rgb key should be the first one owing to the viewer bug 
        return {"rgb": rgb, "virtual_views_img": virtual_views_img, 
            "spike_blur":spike_blur, "virtual_views_recon":virtual_views_recon,
            "virtual_views_tfp":virtual_views_tfp,
            "depth": depth_im, "accumulation": alpha, "background": background,}

    def after_train(self, step: int):
        assert step == self.step
        # to save some training time, we no longer need to update those stats post refinement
        if self.use_3dgs == False and self.use_spike == True:
            self.config.stop_split_at = -1
        if self.step >= self.config.stop_split_at:
            return
        with torch.no_grad():
            # keep track of a moving average of grad norms
            visible_mask = (self.radii > 0).flatten()
            # BAD-Gaussians: use absgrad if enabled
            if self.config.enable_absgrad:
                assert self.xys.absgrad is not None  # type: ignore
                grads = self.xys.absgrad.detach().norm(dim=-1)  # type: ignore
            else:
                assert self.xys.grad is not None
                grads = self.xys.grad.detach().norm(dim=-1)
            # print(f"grad norm min {grads.min().item()} max {grads.max().item()} mean {grads.mean().item()} size {grads.shape}")
            if self.xys_grad_norm is None:
                self.xys_grad_norm = grads
                self.vis_counts = torch.ones_like(self.xys_grad_norm)
            else:
                assert self.vis_counts is not None
                self.vis_counts[visible_mask] = self.vis_counts[visible_mask] + 1
                self.xys_grad_norm[visible_mask] = grads[visible_mask] + self.xys_grad_norm[visible_mask]
            # update the max screen size, as a ratio of number of pixels
            if self.max_2Dsize is None:
                self.max_2Dsize = torch.zeros_like(self.radii, dtype=torch.float32)
            newradii = self.radii.detach()[visible_mask]
            self.max_2Dsize[visible_mask] = torch.maximum(
                self.max_2Dsize[visible_mask],
                newradii / float(max(self.last_size[0], self.last_size[1])),
            )

    @torch.no_grad()
    def get_outputs_for_camera(
            self,
            camera: Cameras,
            obb_box: Optional[OrientedBox] = None,
            mode: TrajSamplingMode = "mid",
            spike: torch.Tensor = torch.zeros(0)
    ) -> Dict[str, torch.Tensor]:
        """Takes in a camera, generates the raybundle, and computes the output of the model.
        Overridden for a camera-based gaussian model.
        """
        assert camera is not None, "must provide camera to gaussian model"
        self.set_crop(obb_box)
        # BAD-Gaussians: camera.to(device) will drop metadata
        metadata = camera.metadata
        camera = camera.to(self.device)
        camera.metadata = metadata
        outs = self.get_outputs(camera, mode=mode,spike = spike)
        return outs  # type: ignore

    def get_loss_l1_ssim(self,img1,img2):
        Ll1 = torch.abs(img1 - img2).mean()
        simloss = 1 - self.ssim(img1.permute(2, 0, 1)[None, ...], img2.permute(2, 0, 1)[None, ...])
        return (1 - self.config.ssim_lambda) * Ll1 + self.config.ssim_lambda * simloss
    
    def get_loss_mse(self,img1,img2):
        return torch.abs(img1 - img2).mean()
    
    def get_loss_dict(self, outputs, batch, metrics_dict=None,step = 0):
        """ Add loss from the spike-net

        Args:
            batch['image']: blurry input. H * W * 1  
            outputs['rgb']: reblur result from the 3DGS. H * W * 1 
            outputs['spike_blur']: reblur result from the spike-net. H * W * 1 
            outputs['virtual_views_img'] : sequence from the 3DGS. 10 * H * W * 1  
            outputs['virtual_views_recon'] : sequence from the spike-net. 10 * H * W * 1  
            outputs['virtual_views_recon_flip'] : sequence from the spike-net with flipped spike input. 10 * H * W * 1   
        Returns:
            _type_: _description_
        """
        #   "rgb": rgb, "virtual_views_img": virtual_views_img, "spike_blur":spike_blur, "virtual_views_recon":virtual_views_recon}

        loss_dict = {}
        if self.use_3dgs == True:
            # 1. scale loss
            if self.config.use_scale_regularization and self.step % 10 == 0:
                scale_exp = torch.exp(self.scales)
                scale_reg = (
                    torch.maximum(
                        scale_exp.amax(dim=-1) / scale_exp.amin(dim=-1),
                        torch.tensor(self.config.max_gauss_ratio),
                    )
                    - self.config.max_gauss_ratio
                    )
                scale_reg = 0.1 * scale_reg.mean()
            else:
                scale_reg = torch.tensor(0.0).to(self.device)
            loss_dict['scale_reg_loss'] = scale_reg
            # 2. 3dgs reblur loss
            loss_dict['3dgs_reblur_loss'] = self.weight_3dgs * self.get_loss_l1_ssim(outputs['rgb'],batch['image'])

        # 3. spike-net reblur loss
        if self.use_spike == True and self.use_multi_reblur == False:
            loss_dict['spike_reblur_loss'] = self.weight_spike * self.get_loss_l1_ssim(outputs['spike_blur'],batch['image'])
            
        elif self.use_spike == True and self.use_multi_reblur == True:
            spike = batch['spike'][...,None]
            recon_sequence = outputs['virtual_views_recon']
            spike_cumsum = torch.cumsum(spike,dim = 0)
            recon_cumsum = torch.cumsum(recon_sequence,dim = 0)
            temp_loss = 0
            iter_idx = 0
            mid_cam = self.num_cam // 2
            for idx in range(self.num_cam // 4,self.num_cam // 2 + 1):
                start_idx = mid_cam - idx
                end_idx = mid_cam + idx
                start_spike = 40 + start_idx * (96 // (self.num_cam - 1))  
                end_spike = 40 + end_idx * (96 // (self.num_cam - 1))  
                spike_blur = (spike_cumsum[end_spike] - spike_cumsum[start_spike - 1]) / (end_spike - start_spike + 1)
                if start_idx == 0:
                    spike_reblur = recon_cumsum[end_idx] / (end_idx + 1)
                else:
                    spike_reblur = (recon_cumsum[end_idx] - recon_cumsum[start_idx - 1]) / (end_idx - start_idx + 1)                
                temp_loss += self.get_loss_mse(spike_blur,spike_reblur)
                iter_idx += 1 
            loss_dict['spike_reblur_loss'] = self.weight_spike * temp_loss / iter_idx

        if self.use_3dgs == True and self.use_spike == True:
            # todo 5. mutual loss: reconstructed sequence order might be different
            if self.use_flip:
                loss_dict['3dgs_spike_loss'] = self.weight_joint * min(self.get_loss_mse(outputs['virtual_views_img'],outputs['virtual_views_recon']),
                                                self.get_loss_mse(torch.flip(outputs['virtual_views_img'],[0]),outputs['virtual_views_recon']))
            else:
                loss_dict['3dgs_spike_loss'] = self.weight_joint * self.get_loss_mse(outputs['virtual_views_img'],outputs['virtual_views_recon'])
                
                
        return loss_dict

    def normal(self,img1,img2,img3):
        img1 = (img1 - img1.min()) / (img1.max() - img1.min())
        img2 = (img2 - img2.min()) / (img2.max() - img2.min())
        img3 = (img3 - img3.min()) / (img3.max() - img3.min())
        return img1,img2,img3
        
    def get_metrics_rgb_spike(self,outputs, batch) -> Dict[str, torch.Tensor]:
        metrics = {}
        rgb = outputs['rgb']
        spike_recon = outputs['spike_blur']
        gt = batch['image']
        # channel change
        rgb = torch.permute(rgb,(2,0,1))[None].clip(0,1)
        gt = torch.permute(gt,(2,0,1))[None].clip(0,1)
        spike_recon = torch.permute(spike_recon,(2,0,1))[None].clip(0,1)

        # normalize
        normal_type = 'double'
        if normal_type == 'normal':
            rgb,gt,spike_recon = self.normal(rgb,gt,spike_recon)
        elif normal_type == 'double':
            rgb = (rgb * 2).clip(0,1)
            spike_recon = (spike_recon * 2).clip(0,1)
            
        # rgb metric
        metrics['rgb_psnr'] = self.psnr(rgb, gt)
        metrics['rgb_ssim'] = self.ssim(rgb, gt)
        metrics['rgb_lpips'] = self.lpips(rgb, gt)
        
        # spike metric
        metrics['spike_psnr'] = self.psnr(spike_recon, gt)
        metrics['spike_ssim'] = self.ssim(spike_recon, gt)
        metrics['spike_lpips'] = self.lpips(spike_recon, gt)
        return metrics
    
    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        metrics_dict = super().get_metrics_dict(outputs, batch)
        # Add metrics from camera optimizer
        self.camera_optimizer.get_metrics_dict(metrics_dict)
        return metrics_dict

    def get_param_groups(self) -> Dict[str, List[torch.nn.Parameter]]:
        if self.use_3dgs == True:
            param_groups = super().get_param_groups()
            self.camera_optimizer.get_param_groups(param_groups=param_groups)
        if self.use_spike == True:
            if self.use_3dgs == False:
                param_groups = {}
            param_groups['spike_net'] = list(self.spike_net.parameters())
        return param_groups
