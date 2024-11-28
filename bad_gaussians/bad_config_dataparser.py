"""
BAD-Gaussians dataparser configs.
"""

from nerfstudio.plugins.registry_dataparser import DataParserSpecification
from bad_gaussians.nerf_studio_dataparser import NerfstudioDataParserConfig

DeblurNerfDataParser = DataParserSpecification(config=NerfstudioDataParserConfig())
