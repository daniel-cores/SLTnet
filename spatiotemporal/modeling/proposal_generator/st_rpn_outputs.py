import itertools
import logging
import numpy as np
import torch
import torch.nn.functional as F
from fvcore.nn import smooth_l1_loss

from detectron2.layers import batched_nms, cat
from detectron2.structures import Boxes, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.modeling.sampling import subsample_labels

logger = logging.getLogger(__name__)

"""
Shape shorthand in this module:

    N: number of input frames
    L: number of feature maps per image on which RPN is run
    A: number of cell anchors (must be the same for all feature maps)
    Hi, Wi: height and width of the i-th feature map
    4: size of the box parameterization

Naming convention:

    objectness: refers to the binary classification of an anchor as object vs. not
    object.

    deltas: refers to the 4-d (dx, dy, dw, dh) deltas that parameterize the box2box
    transform (see :class:`box_regression.Box2BoxTransform`).

    pred_objectness_logits: predicted objectness scores in [-inf, +inf]; use
        sigmoid(pred_objectness_logits) to estimate P(object).

    gt_objectness_logits: ground-truth binary classification labels for objectness

    pred_anchor_deltas: predicted box2box transform deltas

    gt_anchor_deltas: ground-truth box2box transform deltas
"""


def find_top_st_rpn_proposals(
    proposals,
    pred_objectness_logits,
    reference_frame_idx,
    image_size,
    nms_thresh,
    pre_nms_topk,
    post_nms_topk,
    min_box_side_len,
    training,
):
    """
    For each feature map, select the `pre_nms_topk` highest scoring proposals,
    apply NMS, clip proposals, and remove small boxes. Return the `post_nms_topk`
    highest scoring proposals among all the feature maps if `training` is True,
    otherwise, returns the highest `post_nms_topk` scoring proposals for each
    feature map.
    Args:
        proposals (list[Tensor]): A list of L tensors. Tensor i has shape (N, Hi*Wi*A, 4).
            All proposal predictions on the feature maps.
        pred_objectness_logits (list[Tensor]): A list of L tensors. Tensor i has shape (N, Hi*Wi*A).
        reference_frame_idx (int): Reference frame index used to select boxes/scores to execute
            NMS. 
        image_size: Input images size in (h, w) order.
        nms_thresh (float): IoU threshold to use for NMS
        pre_nms_topk (int): number of top k scoring proposals to keep before applying NMS.
            When RPN is run on multiple feature maps (as in FPN) this number is per
            feature map.
        post_nms_topk (int): number of top k scoring proposals to keep after applying NMS.
            When RPN is run on multiple feature maps (as in FPN) this number is total,
            over all feature maps.
        min_box_side_len (float): minimum proposal box side length in pixels (absolute units
            wrt input images).
        training (bool): True if proposals are to be used in training, otherwise False.
            This arg exists only to support a legacy bug; look for the "NB: Legacy bug ..."
            comment.
    Returns:
        proposals (list[Instances]): list of N Instances. The i-th Instances
            stores post_nms_topk object proposals for frame i, sorted by the
            objectness score in the reference frame in descending order.
    """
    device = proposals[0].device

    # 1. Select top-k anchor for every level and every image
    topk_scores = []  # #lvl Tensor, each of shape N x topk
    topk_proposals = []
    level_ids = []  # #lvl Tensor, each of shape (topk,)
    for level_id, proposals_i, logits_i in zip(
        itertools.count(), proposals, pred_objectness_logits
    ):
        Hi_Wi_A = logits_i.shape[1]
        num_proposals_i = min(pre_nms_topk, Hi_Wi_A)

        # sort is faster than topk (https://github.com/pytorch/pytorch/issues/22812)
        # topk_scores_i, topk_idx = logits_i.topk(num_proposals_i, dim=1)
        logits_i, idx = logits_i[reference_frame_idx].sort(descending=True, dim=0)
        topk_scores_i = logits_i[:num_proposals_i]
        topk_idx = idx[:num_proposals_i]
        
        topk_proposals_i = proposals_i[:, topk_idx]  # N x topk x 4

        topk_proposals.append(topk_proposals_i)
        topk_scores.append(topk_scores_i)
        level_ids.append(torch.full((num_proposals_i,), level_id, dtype=torch.int64, device=device))

    # 2. Concat all levels together
    topk_scores = cat(topk_scores, dim=0)
    topk_proposals = cat(topk_proposals, dim=1)
    level_ids = cat(level_ids, dim=0)

    # 3. For the reference frame, run a per-level NMS, and choose topk results for 
    # every input frame.
    st_boxes = []
    
    # TODO: cache valid proposals mask for previous frames
    lvl = level_ids
    valid_mask = torch.isfinite(topk_proposals).all(dim=2).all(dim=0) & torch.isfinite(topk_scores)
    if not valid_mask.all():
        topk_proposals = topk_proposals[:, valid_mask]
        topk_scores = topk_scores[valid_mask]
        lvl = lvl[valid_mask]

    keep = None
    st_boxes = []
    for proposal_boxes_f in topk_proposals:
        boxes = Boxes(proposal_boxes_f)
        boxes.clip(image_size)

        # filter empty boxes
        keep_f = boxes.nonempty(threshold=min_box_side_len)
        keep = keep_f if keep is None else keep & keep_f
        
        st_boxes.append(boxes)

    if keep.sum().item() != len(st_boxes[0]):
        topk_scores, lvl = topk_scores[keep], lvl[keep]

    filtered_st_boxes = []
    for boxes in st_boxes:
        if keep.sum().item() != len(boxes):
            boxes = boxes[keep]

        filtered_st_boxes.append(boxes)

    keep = batched_nms(filtered_st_boxes[reference_frame_idx].tensor, topk_scores, lvl, nms_thresh)
    # In Detectron1, there was different behavior during training vs. testing.
    # (https://github.com/facebookresearch/Detectron/issues/459)
    # During training, topk is over the proposals from *all* images in the training batch.
    # During testing, it is over the proposals for each image separately.
    # As a result, the training behavior becomes batch-dependent,
    # and the configuration "POST_NMS_TOPK_TRAIN" end up relying on the batch size.
    # This bug is addressed in Detectron2 to make the behavior independent of batch size.
    keep = keep[:post_nms_topk]  # keep is already sorted

    scores = topk_scores[keep]
    results = []
    for boxes in filtered_st_boxes:
        res = Instances(image_size)
        res.proposal_boxes = boxes[keep]
        res.objectness_logits = scores
        results.append(res)

    return results