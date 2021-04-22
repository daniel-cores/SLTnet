
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch import nn

from detectron2.layers import ShapeSpec
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou, BoxMode
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry

from detectron2.modeling.backbone.resnet import BottleneckBlock, make_stage
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
from detectron2.modeling.sampling import subsample_labels
from detectron2.modeling.roi_heads.box_head import build_box_head
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers, FastRCNNOutputs
from detectron2.modeling.roi_heads.keypoint_head import build_keypoint_head
from detectron2.modeling.roi_heads.mask_head import build_mask_head
from detectron2.modeling.roi_heads import ROIHeads
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY

from .st_classification import StClassificationOutputLayers
from .st_box_head import build_st_box_head
from .tubelet_aggregation_head import build_tubelet_aggregation_head
from .st_fast_rcnn import StFastRCNNOutputs

from scipy.optimize import linear_sum_assignment

##################################################################################################################
LONG_TERM_REGISTRY_TEST = Registry("LONG_TERM_TEST")
LONG_TERM_REGISTRY_TEST.__doc__ = """
TODO refactor
"""

@LONG_TERM_REGISTRY_TEST.register()
def track_center(long_term_roi_buffer, matches, reference_frame_proposals, kf_idx):
    kf_boxes = Boxes(long_term_roi_buffer[kf_idx])
    trans = reference_frame_proposals.proposal_boxes[matches].get_centers() - kf_boxes.get_centers()
    trans = torch.cat([trans, trans], 1)
    long_term_roi_buffer[kf_idx] += trans

    return long_term_roi_buffer


@LONG_TERM_REGISTRY_TEST.register()
def replace(long_term_roi_buffer, matches, reference_frame_proposals, kf_idx):
    long_term_roi_buffer[kf_idx] = reference_frame_proposals.proposal_boxes[matches].tensor

    return long_term_roi_buffer
##################################################################################################################



@ROI_HEADS_REGISTRY.register()
class StROIHeads(ROIHeads):
    """
    Spatio-temporal RoI head based on StandardROIHeads 
    """

    def __init__(self, cfg, input_shape):
        super(StROIHeads, self).__init__(cfg, input_shape)
        self._init_box_head(cfg, input_shape)
        self._init_tubelet_aggregation_head(cfg)

    def _init_tubelet_aggregation_head(self, cfg):
        self.tubelet_aggregation_head = build_tubelet_aggregation_head(cfg)

    def _init_box_head(self, cfg, input_shape):
        # fmt: off
        pooler_resolution        = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales            = tuple(1.0 / input_shape[k].stride for k in self.in_features)
        sampling_ratio           = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type              = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        st_pooler_type           = cfg.MODEL.SPATIOTEMPORAL.ST_POOLER_TYPE
        self.train_on_pred_boxes = cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES
        self.st_cls              = cfg.MODEL.SPATIOTEMPORAL.ST_CLS
        self.spatial_cls         = cfg.MODEL.SPATIOTEMPORAL.SPATIAL_CLS
        self.longterm_proposals  = cfg.MODEL.SPATIOTEMPORAL.ROI_BOX_HEAD.REF_POST_NMS_TOP_N
        self.st_box_head_name    = cfg.MODEL.SPATIOTEMPORAL.ROI_BOX_HEAD.NAME
        self.long_term           = cfg.MODEL.SPATIOTEMPORAL.LONG_TERM
        self.min_box_side_len    = cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE
        # fmt: on
        self.st_cls_short_term_aggregation = cfg.MODEL.SPATIOTEMPORAL.ST_CLS_SHORT_TERM_AGGREGATION
        self.proposal_tracking = cfg.MODEL.SPATIOTEMPORAL.PROPOSAL_TRACKING
        self.test_tracking_type = cfg.MODEL.SPATIOTEMPORAL.TEST_TRACKING_TYPE

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in self.in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]


        self.long_term_proposal_matcher = Matcher(
            [0.3],  # TODO: config(cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS)
            [0, 1], # TODO: config(cfg.MODEL.ROI_HEADS.IOU_LABELS)
            allow_low_quality_matches=False,
        )


        self.box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

        self.st_box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=st_pooler_type,
        )

        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        if self.st_cls:
            self.st_box_head = build_st_box_head(
                cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
            )
            self.st_cls_predictor = StClassificationOutputLayers(
                self.st_box_head.output_size, self.num_classes
            )
        self.box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )
        self.box_predictor = FastRCNNOutputLayers(
            self.box_head.output_size, self.num_classes, self.cls_agnostic_bbox_reg
        )

        if cfg.MODEL.SPATIOTEMPORAL.FREEZE_SPATIAL_HEAD:
            self.freeze_component(self.box_head)
            self.freeze_component(self.box_predictor)

    def freeze_component(self, model):
        for name, p in model.named_parameters():
            p.requires_grad = False

    @torch.no_grad()
    def label_and_sample_proposals(
        self, proposals: List[Instances], targets: List[Instances], reference_frame_idx: int
    ) -> List[Instances]:
        """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns ``self.batch_size_per_image`` random samples from proposals and groundtruth
        boxes, with a fraction of positives that is no larger than
        ``self.positive_sample_fraction``.
        It computes the filtering indexes using the reference frame and applies the same
        filter to the proposals in the other frames.
        Args:
            See :meth:`ROIHeads.forward`
        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. The reference_frame `Instances` has the following fields:
                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                  then the ground-truth box is random)
                Other fields such as "gt_classes", "gt_masks", that's included in `targets`.
        """
        gt_boxes = [x.gt_boxes for x in targets]
        # Augment proposals with ground-truth boxes.
        # In the case of learned proposals (e.g., RPN), when training starts
        # the proposals will be low quality due to random initialization.
        # It's possible that none of these initial
        # proposals have high enough overlap with the gt objects to be used
        # as positive examples for the second stage components (box head,
        # cls head, mask head). Adding the gt boxes to the set of proposals
        # ensures that the second stage components will have some positive
        # examples from the start of training. For RPN, this augmentation improves
        # convergence and empirically improves box AP on COCO by about 0.5
        # points (under one tested configuration).
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        # for proposals_per_image, targets_per_image in zip(proposals, targets):
        targets_reference_frame = targets[reference_frame_idx]  # == targets_per_image
        proposals_reference_frame = proposals[reference_frame_idx] # == proposals_per_image

        has_gt = len(targets_reference_frame) > 0
        match_quality_matrix = pairwise_iou(
            targets_reference_frame.gt_boxes, proposals_reference_frame.proposal_boxes
        )
        matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
        sampled_idxs, gt_classes = self._sample_proposals(
            matched_idxs, matched_labels, targets_reference_frame.gt_classes
        )

        # Set target attributes of the sampled proposals:
        proposals_reference_frame = proposals_reference_frame[sampled_idxs]
        proposals_reference_frame.gt_classes = gt_classes

        # We index all the attributes of targets that start with "gt_"
        # and have not been added to proposals yet (="gt_classes").
        if has_gt:
            sampled_targets = matched_idxs[sampled_idxs]
            # NOTE: here the indexing waste some compute, because heads
            # like masks, keypoints, etc, will filter the proposals again,
            # (by foreground/background, or number of keypoints in the image, etc)
            # so we essentially index the data twice.
            for (trg_name, trg_value) in targets_reference_frame.get_fields().items():
                if trg_name.startswith("gt_") and not proposals_reference_frame.has(trg_name):
                    proposals_reference_frame.set(trg_name, trg_value[sampled_targets])
        else:
            gt_boxes = Boxes(
                targets_reference_frame.gt_boxes.tensor.new_zeros((len(sampled_idxs), 4))
            )
            proposals_reference_frame.gt_boxes = gt_boxes

        proposals_with_gt = []
        for f in range(len(proposals)):
            # apply the same filtering over the others input frames (building object tubelets)
            proposals_with_gt.append(proposals[f][sampled_idxs])
        # Override reference frame index with labels
        proposals_with_gt[reference_frame_idx] = proposals_reference_frame

        num_bg_samples = (gt_classes == self.num_classes).sum().item()
        num_fg_samples = gt_classes.numel() - num_bg_samples

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", num_fg_samples)
        storage.put_scalar("roi_head/num_bg_samples", num_bg_samples)

        return proposals_with_gt


    @torch.no_grad()
    def label_and_sample_long_term(self, proposals, targets):   
        """
        See :class:`StROIHeads.label_and_sample_proposals`.
        """

        gt_boxes = [x.gt_boxes for x in targets]
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        targets_reference_frame = targets[0]  # == targets_per_image
        proposals_reference_frame = proposals[0] # == proposals_per_image

        num_gts = len(targets_reference_frame)
        match_quality_matrix = pairwise_iou(
            targets_reference_frame.gt_boxes, proposals_reference_frame.proposal_boxes
        )
        matched_idxs, matched_labels = self.long_term_proposal_matcher(match_quality_matrix)

        sampled_idxs = list(range(self.longterm_proposals - num_gts)) + list(range(len(proposals_reference_frame) - num_gts, len(proposals_reference_frame)))
        proposals_reference_frame = proposals_reference_frame[sampled_idxs]

        assert num_gts
        # We index all the attributes of targets that start with "gt_"
        sampled_targets = matched_idxs[sampled_idxs]
        matched_labels = matched_labels[sampled_idxs]
        
        for (trg_name, trg_value) in targets_reference_frame.get_fields().items():
            if trg_name.startswith("gt_") and not proposals_reference_frame.has(trg_name):
                proposals_reference_frame.set(trg_name, trg_value[sampled_targets])

        mask = matched_labels == 0
        proposals_reference_frame.gt_id_track[mask] = -1

        return proposals_reference_frame


    def track_proposals_train(self, long_term_proposal, long_term_feature_buffer, targets):
        track_to_boxes = dict(zip([id_track.item() for id_track in targets.gt_id_track], targets.gt_boxes))
        image_size = targets.image_size

        for f_idx, f_lt_proposal in enumerate(long_term_proposal):
            proposal_boxes = f_lt_proposal.proposal_boxes.tensor
            gt_boxes = f_lt_proposal.gt_boxes.tensor
            missed = 0
            for idx, id_track in enumerate(f_lt_proposal.gt_id_track):
                id_track = id_track.item()
                if id_track != -1:
                    if id_track in track_to_boxes:
                        current_gt_box_xywh = BoxMode.convert(track_to_boxes[id_track].unsqueeze(0), BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)[0]

                        long_term_gt_box_xywh = BoxMode.convert(gt_boxes[idx].unsqueeze(0), BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)[0]
                        proposal_box = BoxMode.convert(proposal_boxes[idx].unsqueeze(0), BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)[0]
                        
                        center_delta = current_gt_box_xywh[:2] - long_term_gt_box_xywh[:2]
                        size_ratios = current_gt_box_xywh[2:] / long_term_gt_box_xywh[2:]

                        proposal_box[:2] += center_delta
                        # proposal_box[2:] *= size_ratios
                        proposal_boxes[idx] = BoxMode.convert(proposal_box.unsqueeze(0), BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)[0]

                    else:
                        missed += 1

            f_lt_proposal.proposal_boxes.clip(image_size)

            # filter empty boxes
            keep = f_lt_proposal.proposal_boxes.nonempty(threshold=self.min_box_side_len)
            if keep.sum().item() != len(f_lt_proposal.proposal_boxes):
                long_term_proposal[f_idx], long_term_feature_buffer[f_idx] = \
                    f_lt_proposal[keep], long_term_feature_buffer[f_idx][keep]

        return long_term_proposal, long_term_feature_buffer


    def track_proposals_test(self, long_term_proposal_buffer, reference_frame_proposals, attention_weights, is_kf):
        if not len(long_term_proposal_buffer):
            return long_term_proposal_buffer

        attention_weights = attention_weights[:150, :-1, :]
        reference_frame_proposals = reference_frame_proposals[:150]

        long_term_conf = [torch.nn.functional.softmax(kf.objectness_logits, dim=0).reshape(1,1,-1) for kf in long_term_proposal_buffer]
        long_term_conf = torch.cat(long_term_conf, 1)

        scores = (torch.nn.functional.softmax(reference_frame_proposals.objectness_logits, dim=0).reshape(-1,1,1)) * torch.exp(attention_weights)

        long_term_roi_buffer = [kf.proposal_boxes.tensor for kf in long_term_proposal_buffer]
        for kf_idx in range(len(long_term_roi_buffer)):

            row_ind, col_ind = linear_sum_assignment(scores[:, kf_idx].cpu().numpy(), maximize=True)
            matches = list(range(len(col_ind)))
            for idx, col in enumerate(col_ind):
                matches[col] = row_ind[idx]

            LONG_TERM_REGISTRY_TEST.get(self.test_tracking_type)(long_term_roi_buffer, matches, reference_frame_proposals, kf_idx)

        for idx in range(len(long_term_proposal_buffer)):
            boxes = Boxes(long_term_roi_buffer[idx])
            long_term_proposal_buffer[idx].proposal_boxes = boxes

        return long_term_proposal_buffer


    def preprocess_lt_support_frames(self, long_term_feature_buffer, long_term_proposal_buffer, long_term_targets, reference_frame_idx):
        with torch.no_grad():
            long_term_pooled_features = []
            out_proposal_buffer = []
            # For each long-term frame
            for lt_f, proposals in enumerate(long_term_proposal_buffer):

                tubelet_features = []
                for f_idx, proposal_f in enumerate(proposals):
                    # TODO: only needed for tracking
                    annotated_proposals = self.label_and_sample_long_term([proposal_f], [long_term_targets[lt_f][f_idx]])
                    if f_idx == reference_frame_idx:
                        out_proposal_buffer.append(annotated_proposals)
                    frame_features = [long_term_feature_buffer[lt_f][f_idx][f] for f in self.in_features]
                    
                    box_features = self.st_box_pooler.forward(frame_features, [annotated_proposals.proposal_boxes])
                    tubelet_features.append(box_features)

                aggregated_features = self.tubelet_aggregation_head(tubelet_features, reference_frame_idx)
                long_term_pooled_features.append(aggregated_features)

        return long_term_pooled_features, out_proposal_buffer


    def forward(
        self,
        images: ImageList,
        short_term_features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        reference_frame_idx: int,
        targets: Optional[List[Instances]] = None,
        long_term_feature_buffer: Optional[List[torch.Tensor]] = None,
        long_term_proposal_buffer = None,  # TODO: type
        is_kf: bool=False
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """

        del images
        if self.training:
            assert targets
            # Label tubelet proposals based on last frame box proposals
            proposals = self.label_and_sample_proposals(proposals, targets, reference_frame_idx)

        if self.training:
            losses = self._forward_box(short_term_features, long_term_feature_buffer, long_term_proposal_buffer, proposals, reference_frame_idx, targets)
            del targets
            return proposals, losses
        else:
            pred_instances, features_cur, rois_cur, long_term_rois = self._forward_box(short_term_features, long_term_feature_buffer, long_term_proposal_buffer, proposals, reference_frame_idx, is_kf=is_kf)
            return pred_instances, features_cur, rois_cur, long_term_rois


    def _forward_box(
        self,
        features: List[Dict[str, torch.Tensor]],
        long_term_feature_buffer: List[torch.Tensor],
        long_term_proposal_buffer,  # TODO: type
        proposals: List[Instances],
        reference_frame_idx: int,
        targets: Optional[List[Instances]] = None,
        is_kf: bool=False
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.
        Args:
            features (dict[str, Tensor]): per frame mapping from feature map names to tensor.
                Similar to in :meth:`ROIHeads.forward`.
            long_term_feature_buffer (List[torch.Tensor]): per-frame long-term roi pooled features
            long_term_roi_buffer (List[torch.Tensor]): per-frame long-term proposal_boxes
            proposals (list[Instances]): the per input frame object proposals
                with the matching ground truth for the reference frame (proposals[reference_frame_idx]).
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".
            reference_frame_idx (int) reference frame index in features and proposals lists
        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """

        tubelet_features = []
        reference_frame_features = [features[reference_frame_idx][f] for f in self.in_features]
        reference_frame_proposals_boxes = [proposals[reference_frame_idx].proposal_boxes]
        reference_frame_proposals = proposals[reference_frame_idx]
        for f_idx, proposal_f in enumerate(proposals):
            frame_features = [features[f_idx][f] for f in self.in_features]
            box_features = self.st_box_pooler.forward(frame_features, [proposal_f.proposal_boxes])
            tubelet_features.append(box_features)

        box_features = self.box_pooler(reference_frame_features, reference_frame_proposals_boxes)
        box_features = self.box_head(box_features)
        pred_class_logits, pred_proposal_deltas = self.box_predictor(box_features)
        del box_features

        assert self.st_cls or self.spatial_cls, "At least one classification method is required." 
        if self.spatial_cls:
            ref_pred_class_logits = pred_class_logits
        else:
            ref_pred_class_logits = None

        features_cur, long_term_curr_proposals = None, None
        st_pred_class_logits = None
        attention_weights = None
        long_term_roi_buffer = None
        if self.st_cls:
            if self.st_cls_short_term_aggregation:
                aggregated_features = self.tubelet_aggregation_head(tubelet_features, reference_frame_idx)
            else:
                aggregated_features = tubelet_features[reference_frame_idx]

            if self.st_box_head_name in ["AttentionFCHead"]:
                if self.proposal_tracking and self.training:
                    tracked_proposals, long_term_feature_buffer = self.track_proposals_train(long_term_proposal_buffer, long_term_feature_buffer, targets[reference_frame_idx])
                    long_term_roi_buffer = [proposals.proposal_boxes.tensor for proposals in tracked_proposals]
                elif self.training:
                    long_term_roi_buffer = [proposals.proposal_boxes.tensor for proposals in long_term_proposal_buffer]
                else:
                    long_term_roi_buffer = [kf.proposal_boxes.tensor for kf in long_term_proposal_buffer]

                aggregated_features, features_cur, attention_weights = self.st_box_head(aggregated_features, reference_frame_proposals_boxes, long_term_feature_buffer, long_term_roi_buffer)
                long_term_curr_proposals = reference_frame_proposals[:75] # TODO: param proposals per keyframe
            else:
                aggregated_features = self.st_box_head(aggregated_features)

            st_pred_class_logits = self.st_cls_predictor(aggregated_features)
            del aggregated_features

        del tubelet_features

        outputs = StFastRCNNOutputs(
            self.box2box_transform,
            ref_pred_class_logits,
            st_pred_class_logits,
            pred_proposal_deltas,
            [proposals[reference_frame_idx]],
            self.smooth_l1_beta,
        )
        if self.training:
            if self.train_on_pred_boxes:
                assert False, 'Not implemented'
            return outputs.losses()
        else:
            pred_instances, kept_indices = outputs.inference(
                self.test_score_thresh, self.test_nms_thresh, self.test_detections_per_img
            )

            if self.proposal_tracking:
                long_term_proposal_buffer = self.track_proposals_test(long_term_proposal_buffer, proposals[reference_frame_idx], attention_weights, is_kf)

            return pred_instances, features_cur, long_term_curr_proposals, long_term_proposal_buffer
