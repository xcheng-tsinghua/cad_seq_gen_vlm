"""Model package — single-view SDXL + ControlNet + IP-Adapter."""

from .pipeline import (
    CADMultiViewPipeline,
    CADPipelineOutput,
    CADSingleViewPipeline,
    ImageProjModel,
    IPAttnProcessor,
)

__all__ = [
    "CADSingleViewPipeline",
    "CADMultiViewPipeline",
    "CADPipelineOutput",
    "ImageProjModel",
    "IPAttnProcessor",
]
