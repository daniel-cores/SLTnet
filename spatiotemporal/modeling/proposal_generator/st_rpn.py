from typing import Dict, List
import torch
import torch.nn.functional as F
from torch import nn

from detectron2.layers import ShapeSpec
from detectron2.structures import ImageList
from detectron2.modeling import build_anchor_generator
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.proposal_generator.build import PROPOSAL_GENERATOR_REGISTRY
from detectron2.modeling.proposal_generator import build_rpn_head
from detectron2.modeling.proposal_generator.rpn_outputs import RPNOutputs
from detectron2.modeling.proposal_generator.rpn_outputs import find_top_rpn_proposals

from .st_rpn_outputs import find_top_st_rpn_proposals

@PROPOSAL_GENERATOR_REGISTRY.register()
class ST_RPN(nn.Module):
    """
    Region Proposal Network, introduced by the Faster R-CNN paper.
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()

        # fmt: off
        self.min_box_side_len        = cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE
        self.in_features             = cfg.MODEL.RPN.IN_FEATURES
        self.nms_thresh              = cfg.MODEL.RPN.NMS_THRESH
        self.batch_size_per_image    = cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE
        self.positive_fraction       = cfg.MODEL.RPN.POSITIVE_FRACTION
        self.smooth_l1_beta          = cfg.MODEL.RPN.SMOOTH_L1_BETA
        self.loss_weight             = cfg.MODEL.RPN.LOSS_WEIGHT
        self.num_frames              = cfg.MODEL.SPATIOTEMPORAL.NUM_FRAMES
        self.temporal_dropout        = cfg.MODEL.SPATIOTEMPORAL.TEMPORAL_DROPOUT
        # fmt: on

        if self.temporal_dropout:
            assert cfg.MODEL.SPATIOTEMPORAL.FORWARD_AGGREGATION, "Temporal dropout without forward aggregation."

        if cfg.MODEL.SPATIOTEMPORAL.FORWARD_AGGREGATION:
            # (f_{t-NUM_FRAMES}, ..., f_{t-1}, f_t, f_{t+1}, ..., f_{t+NUM_FRAMES})
            self.num_frames = (2 * self.num_frames) + 1

        # Map from self.training state to train/test settings
        self.pre_nms_topk = {
            True: cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN,
            False: cfg.MODEL.RPN.PRE_NMS_TOPK_TEST,
        }
        self.post_nms_topk = {
            True: cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN,
            False: cfg.MODEL.RPN.POST_NMS_TOPK_TEST,
        }
        self.boundary_threshold = cfg.MODEL.RPN.BOUNDARY_THRESH

        self.anchor_generator = build_anchor_generator(
            cfg, [input_shape[f] for f in self.in_features]
        )
        self.box2box_transform = Box2BoxTransform(weights=cfg.MODEL.RPN.BBOX_REG_WEIGHTS)
        self.anchor_matcher = Matcher(
            cfg.MODEL.RPN.IOU_THRESHOLDS, cfg.MODEL.RPN.IOU_LABELS, allow_low_quality_matches=True
        )
        self.rpn_head = build_rpn_head(cfg, [input_shape[f] for f in self.in_features])

    # def forward(self, images, features, reference_frame_idx, rpn_outputs_buffer, gt_instances=None):
    def forward(self, images, features, reference_frame_idx, predict_proposals, predict_objectness_logits, gt_instances=None, force_nms=False):
        """
        Args:
            images (ImageList): input images of length `N`
            features (dict[str: Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            reference_frame_idx (int): reference frame index
            predict_proposals (List[Tensor]): predict proposals buffer in neighbour frames used in inference. Tensor per
                feature level with shape [N, k, 4]
            predict_objectness_logits (List[Tensor]): predict objectness buffer in neighbour frames used in inference.
                Tensor per feature level with shape [N, k]
            gt_instances (list[Instances], optional): a length `N` list of `Instances`s.
                Each `Instances` stores ground-truth instances for the corresponding image.
            force_nms (bool): applies find_top_st_rpn_proposals even with less than self.num_frames frame proposals in
                predict_proposals buffer.

        Returns:
            tubelet_proposals: list[Instances]: contains fields "proposal_boxes" (for every input frame), "objectness_logits"
            loss: dict[Tensor] or None
            predict_proposals (List[Tensor]): in inference time, input predict_proposals updated with current
                frame information
            predict_objectness_logits (List[Tensor]): in inference time, input predict_objectness_logits updated with
                current frame information
        """
        gt_boxes = [x.gt_boxes for x in gt_instances] if gt_instances is not None else None
        del gt_instances
        features = [features[f] for f in self.in_features]
        pred_objectness_logits, pred_anchor_deltas = self.rpn_head(features)
        anchors = self.anchor_generator(features)
        # TODO: The anchors only depend on the feature map shape; there's probably
        # an opportunity for some optimizations (e.g., caching anchors).
        outputs = RPNOutputs(
            self.box2box_transform,
            self.anchor_matcher,
            self.batch_size_per_image,
            self.positive_fraction,
            images,
            pred_objectness_logits,
            pred_anchor_deltas,
            anchors,
            self.boundary_threshold,
            gt_boxes,
            self.smooth_l1_beta,
        )

        if self.training:
            losses = {k: v * self.loss_weight for k, v in outputs.losses().items()}

            predict_proposals = outputs.predict_proposals()
            predict_objectness_logits = outputs.predict_objectness_logits()
        else:
            losses = {}

            if predict_proposals is None:
                # Empty buffer
                predict_proposals = outputs.predict_proposals()
                predict_objectness_logits = outputs.predict_objectness_logits()
            else:
                current_predict_proposals = outputs.predict_proposals()
                current_objectness_logits = outputs.predict_objectness_logits()

                for lvl in range(len(predict_proposals)):
                    if predict_proposals[lvl].shape[0] == self.num_frames:
                        # removes oldest frame data
                        # TODO: optimize?
                        predict_proposals[lvl] = predict_proposals[lvl][1:]
                        predict_objectness_logits[lvl] = predict_objectness_logits[lvl][1:]

                    # append new frame data
                    predict_proposals[lvl] = torch.cat(
                        (predict_proposals[lvl], current_predict_proposals[lvl]), 0
                    )
                    predict_objectness_logits[lvl] = torch.cat(
                        (predict_objectness_logits[lvl], current_objectness_logits[lvl]), 0
                    )

        tubelet_proposals = None
        # we need to process self.num_frames to calculate tubelet proposal in test, otherwise return
        # just the complete proposal set for the current frame without NMS. This proposal set will be 
        # used in the next iteration. The force_single_frame param changes this behavior applying the 
        # find_top_st_rpn_proposals to any number of input frames (used in train with temporal dropout).
        if (predict_proposals[0].shape[0] == self.num_frames) or force_nms:
            with torch.no_grad():
                # Find the top proposals by applying NMS and removing boxes that
                # are too small. The proposals are treated as fixed for approximate
                # joint training with roi heads. This approach ignores the derivative
                # w.r.t. the proposal boxes’ coordinates that are also network
                # responses, so is approximate.

                image = ImageList(
                    images[0],
                    [images.image_sizes[0]]
                )
                images_size = images.image_sizes[0]  # in (h, w) order

                tubelet_proposals = find_top_st_rpn_proposals(
                    predict_proposals,
                    predict_objectness_logits,
                    reference_frame_idx,
                    images_size,
                    self.nms_thresh,
                    self.pre_nms_topk[self.training],
                    self.post_nms_topk[self.training],
                    self.min_box_side_len,
                    self.training,
                )

        return tubelet_proposals, losses, predict_proposals, predict_objectness_logits
