"""Image restoration pipeline."""
from __future__ import annotations

import os
from pathlib import Path
from time import time
from typing import Optional, Type

import torch
from dataclasses import dataclass, field
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanager
from nerfstudio.data.datamanagers.parallel_datamanager import ParallelDataManager
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from nerfstudio.utils import profiler
from nerfstudio.utils.writer import to8b
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple, Type, Union, cast
from bad_gaussians.bad_gaussians import BadGaussiansModel

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2

def to8b(x,nor = False):
    """Converts a torch tensor to 8 bit"""
    return (255 * torch.clamp(x, min=0, max=1)).to(torch.uint8)

@dataclass
class ImageRestorationPipelineConfig(VanillaPipelineConfig):
    """Image restoration pipeline config"""

    _target: Type = field(default_factory=lambda: ImageRestorationPipeline)
    """The target class to be instantiated."""

    eval_render_start_end: bool = False
    """whether to render and save the starting and ending virtual sharp images in eval"""

    eval_render_estimated: bool = False
    """whether to render and save the estimated degraded images with learned trajectory in eval.
    Note: Slow & VRAM hungry! Reduce VRAM consumption by passing argument
            `--pipeline.model.eval_num_rays_per_chunk=16384` or less.
    """


class ImageRestorationPipeline(VanillaPipeline):
    """Image restoration pipeline"""

    config: ImageRestorationPipelineConfig

    
    # metrics are not accurate because the batch input is the tfp-blur not the sharp image 
    # results are stored to the tensorboard
    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        """This function gets your training loss dict. This will be responsible for
        getting the next batch of data from the DataManager and interfacing with the
        Model class, feeding the data to the model's forward function.

        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """
        ray_bundle, batch = self.datamanager.next_train(step)
        model_outputs = self._model(camera = ray_bundle,mode = 'uniform', spike = batch['spike'])  # train distributed data parallel model if world_size > 1
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict,step)
        return model_outputs, loss_dict, metrics_dict
    
    # results are stored to the tensorboard [Eval Loss,Eval Loss Dict,Eval Metrics Dict]
    def get_eval_loss_dict(self, step: int) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        self.eval()
        ray_bundle, batch = self.datamanager.next_eval(step)
        model_outputs = self._model(camera = ray_bundle,mode = 'uniform', spike = batch['spike'])  # train distributed data parallel model if world_size > 1
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict,step)
        self.train()
        return model_outputs, loss_dict, metrics_dict
    
    def post_process(self,img,normal_type = "double"):
        if normal_type == "normal":
            return (img - img.min()) / (img.max() - img.min())
        elif normal_type == "double":
            return (img * 2).clip(0,1)
            
        
    @profiler.time_function
    def get_average_eval_image_metrics(
            self, step: Optional[int] = None, output_path: Optional[Path] = None, get_std: bool = False
    ):
        """Iterate over all the images in the eval dataset and get the average.
        Also saves the rendered images to disk if output_path is provided.

        Args:
            step: current training step
            output_path: optional path to save rendered images to
            get_std: Set True if you want to return std with the mean metric.

        Returns:
            metrics_dict: dictionary of metrics
        """
        # save the network 
        image_dir = output_path / f"{step:06}"
        path = str((image_dir / f"0000.pth").resolve())
        if step % 10000 == 0:
            if not image_dir.exists():
                image_dir.mkdir(parents=True)
            self.model.save_spike_net(path)
        # evaluation
        self.eval()
        metrics_dict_list = []
        render_list = ["mid"]
        if self.config.eval_render_start_end:
            render_list += ["start", "end"]
        if self.config.eval_render_estimated:
            render_list += ["uniform"]
        assert isinstance(self.datamanager, (VanillaDataManager, ParallelDataManager, FullImageDatamanager))
        num_images = len(self.datamanager.fixed_indices_eval_dataloader)
        with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeElapsedColumn(),
                MofNCompleteColumn(),
                transient=True,
        ) as progress:
            task = progress.add_task("[green]Evaluating all eval images...", total=num_images)
            iteri = 0
            for camera, batch in self.datamanager.fixed_indices_eval_dataloader:
                # time this the following line
                inner_start = time()
                image_idx = batch['image_idx']
                images_dict = {
                    f"{image_idx:04}_input": batch["degraded"][:, :, :3],
                    f"{image_idx:04}_gt": batch["image"][:, :, :3],
                }
                # mid start end, uniform -> return the re-renderd image
                for mode in render_list:
                    outputs = self.model.get_outputs_for_camera(camera, mode=mode,spike = batch['spike'])
                    # uniform mode: save the reblur image, spike_blur,spike_start and spike_end
                    if mode == "uniform":
                        images_dict[f"{image_idx:04}_estimated"] = self.post_process(outputs["rgb"])
                        images_dict[f"{image_idx:04}_estimated_spike"] = self.post_process(outputs["spike_blur"])
                    else:
                        images_dict[f"{image_idx:04}_rgb_{mode}"] = self.post_process(outputs["rgb"])
                        images_dict[f"{image_idx:04}_spike_{mode}"] = self.post_process(outputs["spike_blur"])
                    if mode == "mid":
                        metrics_dict = self.model.get_metrics_rgb_spike(outputs, batch)
                if step % 10000 == 0 and iteri % 5 == 0:
                    for filename, data in images_dict.items():
                        data = data.detach().cpu()
                        is_u8_image = False
                        for tag in ["rgb", "input", "gt", "estimated", "mask",'spike']:
                            if tag in filename:
                                is_u8_image = True
                        if is_u8_image:
                            path = str((image_dir / f"{filename}.png").resolve())
                            cv2.imwrite(path, cv2.cvtColor(to8b(data,nor = False).numpy(), cv2.COLOR_RGB2BGR))
                        else:
                            path = str((image_dir / f"{filename}.exr").resolve())
                            cv2.imwrite(path, data.numpy())
                assert "num_rays_per_sec" not in metrics_dict
                height, width = camera.height, camera.width
                num_rays = height * width
                metrics_dict["num_rays_per_sec"] = (num_rays / (time() - inner_start)).item()
                fps_str = "fps"
                assert fps_str not in metrics_dict
                metrics_dict[fps_str] = (metrics_dict["num_rays_per_sec"] / (height * width)).item()
                metrics_dict_list.append(metrics_dict)
                progress.advance(task)
                iteri += 1
        # average the metrics list
        metrics_dict = {}
        for key in metrics_dict_list[0].keys():
            if get_std:
                key_std, key_mean = torch.std_mean(
                    torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list])
                )
                metrics_dict[key] = float(key_mean)
                metrics_dict[f"{key}_std"] = float(key_std)
            else:
                metrics_dict[key] = float(
                    torch.mean(torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list]))
                )
        self.train()
        return metrics_dict
