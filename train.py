# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python
"""Train a radiance field with nerfstudio.
For real captures, we recommend using the [bright_yellow]nerfacto[/bright_yellow] model.

Nerfstudio allows for customizing your training and eval configs from the CLI in a powerful way, but there are some
things to understand.

The most demonstrative and helpful example of the CLI structure is the difference in output between the following
commands:

    ns-train -h
    ns-train nerfacto -h nerfstudio-data
    ns-train nerfacto nerfstudio-data -h

In each of these examples, the -h applies to the previous subcommand (ns-train, nerfacto, and nerfstudio-data).

In the first example, we get the help menu for the ns-train script.
In the second example, we get the help menu for the nerfacto model.
In the third example, we get the help menu for the nerfstudio-data dataparser.

With our scripts, your arguments will apply to the preceding subcommand in your command, and thus where you put your
arguments matters! Any optional arguments you discover from running

    ns-train nerfacto -h nerfstudio-data

need to come directly after the nerfacto subcommand, since these optional arguments only belong to the nerfacto
subcommand:

    ns-train nerfacto {nerfacto optional args} nerfstudio-data
"""

from __future__ import annotations
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import random
import socket
import traceback
from datetime import timedelta
from typing import Any, Callable, Literal, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import tyro
import yaml

from nerfstudio.configs.config_utils import convert_markup_to_ansi
from nerfstudio.configs.method_configs import AnnotatedBaseConfigUnion
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.utils import comms, profiler
from nerfstudio.utils.rich_utils import CONSOLE

DEFAULT_TIMEOUT = timedelta(minutes=30)

# speedup for when input size to model doesn't change (much)
torch.backends.cudnn.benchmark = True  # type: ignore


def _ensure_cuda_toolkit_visible() -> None:
    """Ensure gsplat can find CUDA toolkit (nvcc) for JIT compile.

    gsplat (0.1.x) uses torch cpp_extension JIT and checks for CUDA toolkit via
    CUDA_HOME and/or nvcc in PATH. When mp.spawn uses 'spawn', child processes
    may miss shell-initialized PATH/CUDA_HOME; we set them here in-process.
    """
    cuda_candidates = ("/usr/local/cuda", "/usr/local/cuda-12.4", "/usr/local/cuda-12")
    if "CUDA_HOME" not in os.environ or not os.environ.get("CUDA_HOME"):
        for p in cuda_candidates:
            if os.path.exists(p):
                os.environ["CUDA_HOME"] = p
                break
    cuda_bin = os.path.join(os.environ.get("CUDA_HOME", "/usr/local/cuda"), "bin")
    path = os.environ.get("PATH", "")
    if os.path.exists(cuda_bin) and cuda_bin not in path.split(":"):
        os.environ["PATH"] = f"{cuda_bin}:{path}" if path else cuda_bin


def _warmup_gsplat_cuda() -> None:
    """Preload/compile gsplat CUDA extension in the parent process.

    This avoids multi-process races around torch_extensions build locks.
    """
    _ensure_cuda_toolkit_visible()
    try:
        from gsplat.cuda._backend import _C  # noqa: F401
        if _C is None:
            # Keep message actionable: this is the exact failure users see later.
            raise RuntimeError(
                "gsplat CUDA backend is None (CUDA toolkit not found or JIT compile failed). "
                f"CUDA_HOME={os.environ.get('CUDA_HOME')} PATH_has_nvcc={'nvcc' in os.environ.get('PATH','')}"
            )
    except Exception as e:
        CONSOLE.log(f"[red]gsplat warmup failed:[/red] {e}")
        raise


def _find_free_port() -> str:
    """Finds a free port."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _set_random_seed(seed) -> None:
    """Set randomness seed in torch and numpy"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(seed)


def train_loop(local_rank: int, world_size: int, config: TrainerConfig, global_rank: int = 0):
    """Main training function that sets up and runs the trainer per process

    Args:
        local_rank: current rank of process
        world_size: total number of gpus available
        config: config file specifying training regimen
    """
    global opt
    config.machine.seed = config.machine.seed + global_rank + cfg.seed_set
    _set_random_seed(config.machine.seed)
    trainer = config.setup(local_rank=local_rank, world_size=world_size)
    trainer.setup()
    trainer.train()


def _distributed_worker(
    local_rank: int,
    main_func: Callable,
    world_size: int,
    num_devices_per_machine: int,
    machine_rank: int,
    dist_url: str,
    config: TrainerConfig,
    timeout: timedelta = DEFAULT_TIMEOUT,
    device_type: Literal["cpu", "cuda", "mps"] = "cuda",
) -> Any:
    """Spawned distributed worker that handles the initialization of process group and handles the
       training process on multiple processes.

    Args:
        local_rank: Current rank of process.
        main_func: Function that will be called by the distributed workers.
        world_size: Total number of gpus available.
        num_devices_per_machine: Number of GPUs per machine.
        machine_rank: Rank of this machine.
        dist_url: URL to connect to for distributed jobs, including protocol
            E.g., "tcp://127.0.0.1:8686".
            It can be set to "auto" to automatically select a free port on localhost.
        config: TrainerConfig specifying training regimen.
        timeout: Timeout of the distributed workers.

    Raises:
        e: Exception in initializing the process group

    Returns:
        Any: TODO: determine the return type
    """
    assert torch.cuda.is_available(), "cuda is not available. Please check your installation."
    global_rank = machine_rank * num_devices_per_machine + local_rank

    dist.init_process_group(
        backend="nccl" if device_type == "cuda" else "gloo",
        init_method=dist_url,
        world_size=world_size,
        rank=global_rank,
        timeout=timeout,
    )
    assert comms.LOCAL_PROCESS_GROUP is None
    num_machines = world_size // num_devices_per_machine
    for i in range(num_machines):
        ranks_on_i = list(range(i * num_devices_per_machine, (i + 1) * num_devices_per_machine))
        pg = dist.new_group(ranks_on_i)
        if i == machine_rank:
            comms.LOCAL_PROCESS_GROUP = pg

    assert num_devices_per_machine <= torch.cuda.device_count()
    output = main_func(local_rank, world_size, config, global_rank)
    comms.synchronize()
    dist.destroy_process_group()
    return output


def launch(
    main_func: Callable,
    num_devices_per_machine: int,
    num_machines: int = 1,
    machine_rank: int = 0,
    dist_url: str = "auto",
    config: Optional[TrainerConfig] = None,
    timeout: timedelta = DEFAULT_TIMEOUT,
    device_type: Literal["cpu", "cuda", "mps"] = "cuda",
) -> None:
    """Function that spawns multiple processes to call on main_func

    Args:
        main_func (Callable): function that will be called by the distributed workers
        num_devices_per_machine (int): number of GPUs per machine
        num_machines (int, optional): total number of machines
        machine_rank (int, optional): rank of this machine.
        dist_url (str, optional): url to connect to for distributed jobs.
        config (TrainerConfig, optional): config file specifying training regimen.
        timeout (timedelta, optional): timeout of the distributed workers.
        device_type: type of device to use for training.
    """
    assert config is not None
    world_size = num_machines * num_devices_per_machine
    if world_size == 0:
        raise ValueError("world_size cannot be 0")
    elif world_size == 1:
        # uses one process
        try:
            main_func(local_rank=0, world_size=world_size, config=config)
        except KeyboardInterrupt:
            # print the stack trace
            CONSOLE.print(traceback.format_exc())
        finally:
            profiler.flush_profiler(config.logging)
    elif world_size > 1:
        # Using multiple gpus with multiple processes.
        if dist_url == "auto":
            assert num_machines == 1, "dist_url=auto is not supported for multi-machine jobs."
            port = _find_free_port()
            dist_url = f"tcp://127.0.0.1:{port}"
        if num_machines > 1 and dist_url.startswith("file://"):
            CONSOLE.log("file:// is not a reliable init_method in multi-machine jobs. Prefer tcp://")

        process_context = mp.spawn(
            _distributed_worker,
            nprocs=num_devices_per_machine,
            join=False,
            args=(main_func, world_size, num_devices_per_machine, machine_rank, dist_url, config, timeout, device_type),
        )
        # process_context won't be None because join=False, so it's okay to assert this
        # for Pylance reasons
        assert process_context is not None
        try:
            process_context.join()
        except KeyboardInterrupt:
            for i, process in enumerate(process_context.processes):
                if process.is_alive():
                    CONSOLE.log(f"Terminating process {i}...")
                    process.terminate()
                process.join()
                CONSOLE.log(f"Process {i} finished.")
        finally:
            profiler.flush_profiler(config.logging)


def main(config: TrainerConfig) -> None:
    """Main function."""
    
    if config.data:
        CONSOLE.log("Using --data alias for --data.pipeline.datamanager.data")
        config.pipeline.datamanager.data = config.data

    if config.prompt:
        CONSOLE.log("Using --prompt alias for --data.pipeline.model.prompt")
        config.pipeline.model.prompt = config.prompt

    if config.load_config:
        CONSOLE.log(f"Loading pre-set config from: {config.load_config}")
        config = yaml.load(config.load_config.read_text(), Loader=yaml.Loader)

    config.set_timestamp()

    # print and save config
    config.print_to_terminal()
    config.save_config()

    _warmup_gsplat_cuda()

    launch(
        main_func=train_loop,
        num_devices_per_machine=config.machine.num_devices,
        device_type=config.machine.device_type,
        num_machines=config.machine.num_machines,
        machine_rank=config.machine.machine_rank,
        dist_url=config.machine.dist_url,
        config=config,
    )


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    # Choose a base configuration and override values.
    tyro.extras.set_accent_color("bright_yellow")
    main(
        tyro.cli(
            AnnotatedBaseConfigUnion,
            description=convert_markup_to_ansi(__doc__),
        )
    )
from pathlib import Path
import argparse
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
from dataclasses import dataclass

@dataclass
class TrainConfig:
    seed_set: int = 425 
    use_3dgs: bool = True
    use_spike: bool = True
    use_flip: bool = True
    use_multi_net: bool = True
    use_multi_reblur: bool = True
    use_depth: bool = False
    use_real: bool = False
    data_path: str = 'data/synthetic/wine'
    data_name: str = 'test'
    exp_name: str = 'test'
    weight_3dgs: float = 1
    weight_spike: float = 1
    weight_joint: float = 1
    net_lr: float = 1e-3
    num_cam: int = 13

if __name__ == "__main__":
    cfg = tyro.cli(TrainConfig)
    method_config = ImageRestorationTrainerConfig(
        method_name="bad-gaussians",
        output_dir = Path(f"outputs/{cfg.data_name}"),
        project_name="usp-gaussian",
        experiment_name = cfg.exp_name,
        steps_per_eval_image=3000,
        steps_per_eval_batch=3000,
        steps_per_save=2000,
        steps_per_eval_all_images=3000,
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
                    data = Path(cfg.data_path),
                    load_3D_points=True,
                    eval_mode="fraction",
                ),
                use_real = cfg.use_real
            ),
            model=BadGaussiansModelConfig(
                camera_optimizer=BadCameraOptimizerConfig(mode="linear", num_virtual_views=cfg.num_cam),
                use_scale_regularization=True,
                continue_cull_post_densification=False,
                cull_alpha_thresh=5e-3,
                densify_grad_thresh=4e-4,
                num_downscales=0,
                resolution_schedule=250,
                tv_loss_lambda=None,
                use_spike = cfg.use_spike,
                use_3dgs = cfg.use_3dgs,
                use_flip = cfg.use_flip,
                use_depth = cfg.use_depth,
                use_multi_net = cfg.use_multi_net,
                use_multi_reblur = cfg.use_multi_reblur,
                weight_3dgs = cfg.weight_3dgs,
                weight_spike = cfg.weight_spike,
                weight_joint = cfg.weight_joint
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
                "optimizer": AdamOptimizerConfig(lr=cfg.net_lr, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-5,
                    max_steps=30000,
                ),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15,
                            quit_on_train_completion = True),
        vis="viewer+tensorboard",
    )
    main(method_config)
