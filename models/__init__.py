"""Model package: multi-view ControlNet + custom Diffusion pipeline."""

from .mv_controlnet import (
    CrossViewAttention,
    MultiViewConditioningEmbedding,
    MultiViewControlNetModel,
)
from .pipeline import (
    CADMultiViewPipeline,
    ImageProjModel,
    IPAttnProcessor,
)

__all__ = [
    "CrossViewAttention",
    "MultiViewConditioningEmbedding",
    "MultiViewControlNetModel",
    "CADMultiViewPipeline",
    "ImageProjModel",
    "IPAttnProcessor",
]
