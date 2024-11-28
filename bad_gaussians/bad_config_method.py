"""
BAD-Gaussians configs.
"""

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.plugins.types import MethodSpecification

from bad_gaussians.bad_camera_optimizer import BadCameraOptimizerConfig
from bad_gaussians.bad_gaussians import BadGaussiansModelConfig
from bad_gaussians.image_restoration_full_image_datamanager import ImageRestorationFullImageDataManagerConfig
from bad_gaussians.image_restoration_pipeline import ImageRestorationPipelineConfig
from bad_gaussians.image_restoration_trainer import ImageRestorationTrainerConfig
from bad_gaussians.nerf_studio_dataparser import NerfstudioDataParserConfig

from pathlib import Path

bad_gaussians = MethodSpecification(
    ImageRestorationTrainerConfig(
        method_name="bad-gaussians",
        steps_per_eval_image=3000,
        steps_per_eval_batch=3000,
        steps_per_save=2000,
        steps_per_eval_all_images=1000,
        max_num_iterations=30001,
        mixed_precision=False,
        use_grad_scaler=False,
        gradient_accumulation_steps={"camera_opt": 25},
        pipeline=ImageRestorationPipelineConfig(
            eval_render_start_end=True,
            eval_render_estimated=True,
            datamanager=ImageRestorationFullImageDataManagerConfig(
                cache_images="gpu",  # reduce CPU usage, caused by pin_memory()?
                dataparser=NerfstudioDataParserConfig(
                    load_3D_points=True,
                    eval_mode="fraction",
                ),
            ),
            model=BadGaussiansModelConfig(
                camera_optimizer=BadCameraOptimizerConfig(mode="linear", num_virtual_views=13),
                use_scale_regularization=True,
                continue_cull_post_densification=False,
                cull_alpha_thresh=5e-3,
                densify_grad_thresh=4e-4,
                num_downscales=0,
                resolution_schedule=250,
                tv_loss_lambda=None,
                weight_3dgs = 1,
                weight_spike = 1,
                weight_joint = 1
            ),
        ),
        optimizers={
            "means": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.6e-6,
                    max_steps=30000,
                ),
            },
            "features_dc": {
                "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
                "scheduler": None,
            },
            "features_rest": {
                "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
                "scheduler": None,
            },
            "opacities": {
                "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
                "scheduler": None,
            },
            "scales": {
                "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
                "scheduler": None,
            },
            "quats": {
                "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
                "scheduler": None
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-5,
                    max_steps=30000,
                ),
            },
            "spike_net": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-5,
                    max_steps=30000,
                ),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15,
                            quit_on_train_completion = True),
        vis="viewer+tensorboard",
    ),
    description= 'bad-gauss'
)