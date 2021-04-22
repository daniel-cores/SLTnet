import numpy as np
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.utils.registry import Registry

TUBELET_AGGREGATION_HEAD_REGISTRY = Registry("ROI_BOX_HEAD")
TUBELET_AGGREGATION_HEAD_REGISTRY.__doc__ = """
Registry for tubelet aggregation heads, which summarizes spatio-temporal information.

The registered object will be called with `obj(cfg)`.
"""


"""
Shape shorthand in this module:

    N: number of input frames
    B: number of BBox proposals (must be the same for all input frames)
    H, W: height and width of the feature map (must be the same for all input frames)
    C: number of channels of feature map (must be the same for all input frames)
"""


@TUBELET_AGGREGATION_HEAD_REGISTRY.register()
class MaxPoolAggregationHead(nn.Module):
    """
    Spatio-temporal Max Pooling
    """

    def __init__(self, cfg):
        super().__init__()

    def forward(self, xs, reference_frame_idx):
        x = torch.stack(xs)
        x, _ = torch.max(x, dim=0)
        return x


@TUBELET_AGGREGATION_HEAD_REGISTRY.register()
class AdaptiveWeightAggregationHead(nn.Module):
    """
    Adaptive weight based on FGFA
    """

    def __init__(self, cfg):
        super().__init__()


    def compute_weight(self, tubelet_feat, current_conv_feat):
        tubelet_norm = F.normalize(tubelet_feat, p=2, dim=2)
        current_norm = F.normalize(current_conv_feat, p=2, dim=2)
        weight = torch.sum(tubelet_norm * current_norm, dim=2, keepdims=True)

        return weight


    def forward(self, xs, reference_frame_idx):
        """
        Aggregates features following the FGFA strategy. The calculated weights are
        shared among all channels for the same frame.
        Args:
            xs (list[tensor]) with feature maps for RPN proposals in each frame
            reference_frame_idx (int) reference frame index in xs
        Retruns:
            the aggregated feature map
        """
        x = torch.stack(xs)  # shape: [N, B, C, H, W]
        cur_feat =  x[reference_frame_idx] # shape: [B, C, H, W]
        cur_feat = cur_feat.unsqueeze(0)  # shape: [1, B, C, H, W]

        # Warning: do not modify cur_feat. expand does not make copies of the underling data
        cur_feat = cur_feat.expand(len(xs), -1, -1, -1, -1) # shape: [N, B, C, H, W]

        tubelets_feat = x.permute(1, 0, 2, 3, 4)  # shape: [B, N, C, H, W]
        cur_feat = cur_feat.permute(1, 0, 2, 3, 4)  # shape: [B, N, C, H, W]

        unnormalize_weight = self.compute_weight(tubelets_feat, cur_feat)
        weight = F.softmax(unnormalize_weight, dim=1)  # shape: [B, N, 1, H, W]
        weight = weight.expand(-1, -1, tubelets_feat.shape[2], -1, -1)  # shape: [B, N, C, H, W]

        aggregated_feat = torch.sum(weight * tubelets_feat, dim=1) # shape: [B, C, H, W]

        return aggregated_feat


def build_tubelet_aggregation_head(cfg):
    """
    Build a box head defined by `cfg.MODEL.TUBELET_AGGREGATION_HEAD.NAME`.
    """
    name = cfg.MODEL.TUBELET_AGGREGATION_HEAD.NAME
    return TUBELET_AGGREGATION_HEAD_REGISTRY.get(name)(cfg)