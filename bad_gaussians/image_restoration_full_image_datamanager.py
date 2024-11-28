"""
Full image datamanager for image restoration.
"""
from __future__ import annotations

import random
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Type, Union, cast, Tuple, Dict
from functools import cached_property
from nerfstudio.data.datasets.base_dataset import InputDataset

import torch

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.datamanagers.base_datamanager import variable_res_collate,TDataset
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanager, FullImageDatamanagerConfig
from nerfstudio.data.utils.nerfstudio_collate import nerfstudio_collate
from nerfstudio.utils.rich_utils import CONSOLE
from bad_gaussians.spike_utils import load_vidar_dat
from bad_gaussians.image_restoration_dataloader import ImageRestorationFixedIndicesEvalDataloader, ImageRestorationRandIndicesEvalDataloader
from bad_gaussians.nerf_studio_dataparser import SpikeDataparserOutputs

class SpikeInputDataset(InputDataset):
    def __init__(self, dataparser_outputs: SpikeDataparserOutputs, scale_factor: float = 1.0,split = 'train',use_real = False):
        self.split = split
        self.use_real = use_real
        super().__init__(dataparser_outputs, scale_factor)
    
    def get_data(self, image_idx: int, image_type: Literal["uint8", "float32"] = "float32") -> Dict:
        data = {"image_idx": image_idx}
        # spike reading
        image_idx = data['image_idx']
        spike_filenames = self._dataparser_outputs.spike_filenames
        if self.use_real == False:
            spike = torch.from_numpy(load_vidar_dat(spike_filenames[image_idx],width=600,height=400))
        else:
            spike = torch.from_numpy(load_vidar_dat(spike_filenames[image_idx],width=400,height=250))
            spike = spike[spike.shape[0] // 2 - 88 : spike.shape[0] // 2 + 89]
        data.update({"spike":spike})   
        input_type = "tfp"
        # update sharp images for test dataset
        if self.split == 'train':
            # blurry input from the long tfp
            if input_type == 'tfp':
                tfp_long = torch.mean(data['spike'][40:-40].float(),dim = 0,keepdim = False)[...,None].repeat(1,1,3)
                data.update({"image":tfp_long})
            # blurry input from the rgb image
            elif input_type == 'blur':
                image = self.get_image_float32(image_idx)
                weights = torch.tensor([0.2989, 0.5870, 0.1140], dtype=torch.float32).view(1, 1, 3)
                image = torch.sum(image * weights,dim = -1,keepdim=True).repeat(1,1,3)
                data.update({"image":image})
        elif self.split == 'test':
            image = self.get_image_float32(image_idx)
            weights = torch.tensor([0.2989, 0.5870, 0.1140], dtype=torch.float32).view(1, 1, 3)
            image = torch.sum(image * weights,dim = -1,keepdim=True).repeat(1,1,3)
            data.update({"image":image})
        return data
    
    
    
@dataclass
class ImageRestorationFullImageDataManagerConfig(FullImageDatamanagerConfig):
    """Datamanager for image restoration"""

    _target: Type = field(default_factory=lambda: ImageRestorationFullImageDataManager)
    """Target class to instantiate."""
    collate_fn: Callable[[Any], Any] = cast(Any, staticmethod(nerfstudio_collate))
    """Specifies the collate function to use for the train and eval dataloaders."""
    use_real: bool = False
    """Input is real-world dataset"""

class ImageRestorationFullImageDataManager(FullImageDatamanager):  # pylint: disable=abstract-method
    """Data manager implementation for image restoration
    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: ImageRestorationFullImageDataManagerConfig

    def __init__(
            self,
            config: ImageRestorationFullImageDataManagerConfig,
            device: Union[torch.device, str] = "cpu",
            test_mode: Literal["test", "val", "inference"] = "val",
            world_size: int = 1,
            local_rank: int = 0,
            **kwargs,
    ):
        super().__init__(config, device, test_mode, world_size, local_rank, **kwargs)
  

        self._fixed_indices_eval_dataloader = ImageRestorationFixedIndicesEvalDataloader(
            input_dataset=self.eval_dataset,
            degraded_dataset=self.train_dataset,
            device=self.device,
            num_workers=self.world_size * 4,
        )
        self.eval_dataloader = ImageRestorationRandIndicesEvalDataloader(
            input_dataset=self.eval_dataset,
            degraded_dataset=self.train_dataset,
            device=self.device,
            num_workers=self.world_size * 4,
        )

    @property
    def fixed_indices_eval_dataloader(self):
        """Returns the fixed indices eval dataloader"""
        return self._fixed_indices_eval_dataloader
    
    def next_train(self, step: int) -> Tuple[Cameras, Dict]:
        """Returns the next training batch with the spike

        Returns a Camera instead of raybundle"""
        camera, data = super().next_train(step)
        # load the spike to float and cuda
        data["spike"] = data["spike"].float().to(self.device) 
        return camera, data
    
    def next_eval(self, step: int) -> Tuple[Cameras, Dict]:
        """Returns the next evaluation batch. Returns a Camera instead of raybundle"""
        image_idx = self.eval_unseen_cameras.pop(random.randint(0, len(self.eval_unseen_cameras) - 1))
        # Make sure to re-populate the unseen cameras list if we have exhausted it
        if len(self.eval_unseen_cameras) == 0:
            self.eval_unseen_cameras = [i for i in range(len(self.eval_dataset))]
        data = deepcopy(self.cached_eval[image_idx])
        data["image"] = data["image"].to(self.device)
        data["spike"] = data["spike"].float().to(self.device) 
        assert len(self.eval_dataset.cameras.shape) == 1, "Assumes single batch dimension"
        camera = self.eval_dataset.cameras[image_idx : image_idx + 1].to(self.device)
        # BAD-Gaussians: pass camera index to BadNerfCameraOptimizer
        if camera.metadata is None:
            camera.metadata = {}
        camera.metadata["cam_idx"] = image_idx
        return camera, data

    def next_eval_image(self, step: int) -> Tuple[Cameras, Dict]:
        for camera, batch in self.eval_dataloader:
            assert camera.shape[0] == 1
            return camera, batch
        raise ValueError("No more eval images")

    @cached_property
    def dataset_type(self) -> Type[TDataset]:
        return SpikeInputDataset
    
    def create_train_dataset(self) -> TDataset:
        """Sets up the data loaders for training"""
        return self.dataset_type(
            dataparser_outputs=self.train_dataparser_outputs,
            scale_factor=self.config.camera_res_scale_factor,
            split = 'train',
            use_real = self.config.use_real
        )
        
    def create_eval_dataset(self) -> TDataset:
        """Sets up the data loaders for evaluation"""
        return self.dataset_type(
            dataparser_outputs=self.dataparser.get_dataparser_outputs(split='test'),
            scale_factor=self.config.camera_res_scale_factor,
            split = 'test',
            use_real = self.config.use_real
        )