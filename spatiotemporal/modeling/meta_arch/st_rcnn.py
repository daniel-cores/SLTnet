import logging
import numpy as np
import torch
import random
from torch import nn

from collections import deque

from detectron2.structures import ImageList
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n

from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.roi_heads import build_roi_heads
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY


__all__ = ["GeneralizedST_RCNN"]


@META_ARCH_REGISTRY.register()
class GeneralizedST_RCNN(nn.Module):
    """
    Generalized spatio-temporal R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)
        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())
        self.roi_heads = build_roi_heads(cfg, self.backbone.output_shape())
        self.vis_period = cfg.VIS_PERIOD
        self.input_format = cfg.INPUT.FORMAT
        self.current_video = None
        self.frame_idx = 0

        if cfg.MODEL.SPATIOTEMPORAL.FREEZE_BACKBONE:
            self.freeze_component(self.backbone)

        if cfg.MODEL.SPATIOTEMPORAL.FREEZE_PROPOSAL_GENERATOR:
            self.freeze_component(self.proposal_generator)

        self.long_term = cfg.MODEL.SPATIOTEMPORAL.LONG_TERM
        self.temporal_dropout = cfg.MODEL.SPATIOTEMPORAL.TEMPORAL_DROPOUT
        self.num_frames = cfg.MODEL.SPATIOTEMPORAL.NUM_FRAMES
        self.num_keyframes = cfg.MODEL.SPATIOTEMPORAL.NUM_KEYFRAMES
        self.keyframe_interval = cfg.MODEL.SPATIOTEMPORAL.KEYFRAME_INTERVAL
        self.reference_frame_idx = -1

        if cfg.MODEL.SPATIOTEMPORAL.FORWARD_AGGREGATION:
            # (f_{t-NUM_FRAMES}, ..., f_{t-1}, f_t, f_{t+1}, ..., f_{t+NUM_FRAMES})
            self.num_frames = (2 * self.num_frames) + 1
            self.reference_frame_idx = cfg.MODEL.SPATIOTEMPORAL.NUM_FRAMES

        if self.temporal_dropout:
            assert cfg.MODEL.SPATIOTEMPORAL.FORWARD_AGGREGATION, "Temporal dropout without forward aggregation."
        
        if self.temporal_dropout:
            self.reference_frame_idx = cfg.MODEL.SPATIOTEMPORAL.NUM_FRAMES
            self.train_reference_frame_idx = 1
        else:
            self.train_reference_frame_idx = self.reference_frame_idx

        self.short_term_feature_buffer = deque(maxlen=self.num_frames)
        self.long_term_feature_buffer = deque(maxlen=self.num_keyframes)
        self.long_term_roi_buffer = deque(maxlen=self.num_keyframes)
        # RPN buffers
        self.predict_proposals = None
        self.predict_objectness_logits = None

        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        num_channels = len(cfg.MODEL.PIXEL_MEAN)
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(num_channels, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(num_channels, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def freeze_component(self, model):
        for name, p in model.named_parameters():
            p.requires_grad = False

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        def temporal_dropout(inputs):
            previous = random.randint(0, self.reference_frame_idx-1)
            following = random.randint(self.reference_frame_idx+1, self.num_frames-1)
            mask = [previous, self.reference_frame_idx, following]

            return [inputs[i] for i in mask]

        def to_st_features(features):
            # features ([N, C, H, W] per level) to st features list with [1, C, H, W] per level
            st_features = []
            for f_idx in range(len(batched_inputs)):
                lvl_features = {}
                for lvl in features:
                    lvl_features[lvl] = features[lvl][f_idx]
                    lvl_features[lvl] = torch.unsqueeze(lvl_features[lvl], 0)
                st_features.append(lvl_features)
            return st_features


        if not self.training:
            return self.inference(batched_inputs)

        if self.long_term:
            batched_inputs, lt1_batched_inputs, lt2_batched_inputs = \
                batched_inputs[:self.num_frames], batched_inputs[self.num_frames:self.num_frames*2], batched_inputs[-self.num_frames:]

        assert (
            len(batched_inputs) == self.num_frames
        ), "Bad batch size ({}).".format(len(batched_inputs))

        if self.temporal_dropout:
            # TODO: optimization, temporal dropout before loading images
            batched_inputs = temporal_dropout(batched_inputs)
            if self.long_term:
                lt1_batched_inputs = temporal_dropout(lt1_batched_inputs)
                lt2_batched_inputs = temporal_dropout(lt2_batched_inputs)

        images = self.preprocess_image(batched_inputs)
        if self.long_term:
            lt1_images = self.preprocess_image(lt1_batched_inputs)
            lt2_images = self.preprocess_image(lt2_batched_inputs)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            if self.long_term:
                lt1_gt_instances = [x["instances"].to(self.device) for x in lt1_batched_inputs]
                lt2_gt_instances = [x["instances"].to(self.device) for x in lt2_batched_inputs]
        elif "targets" in batched_inputs[0]:
            raise NotImplementedError()
            # log_first_n(
            #     logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            # )
            # gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            raise NotImplementedError("We need gt_instances in training.")
            # gt_instances = None

        features = self.backbone(images.tensor)
        if self.long_term:
            lt1_features = self.backbone(lt1_images.tensor)
            lt2_features = self.backbone(lt2_images.tensor)

        if self.proposal_generator:
            tubelet_proposals, proposal_losses, _, _ = self.proposal_generator(images, features, self.train_reference_frame_idx, None, None, gt_instances, force_nms=True)
            
            if self.long_term:
                # longt-term support frames
                lt1_tubelet_proposals, lt1_proposal_losses, _, _ = self.proposal_generator(
                    lt1_images, lt1_features, self.train_reference_frame_idx, None, None, lt1_gt_instances, force_nms=True
                )
                lt2_tubelet_proposals, lt2_proposal_losses, _, _ = self.proposal_generator(
                    lt2_images, lt2_features, self.train_reference_frame_idx, None, None, lt2_gt_instances, force_nms=True
                )

        else:
            raise NotImplementedError("Precomputed proposal set not supported yet.")
            # assert "proposals" in batched_inputs[0]
            # proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            # proposal_losses = {}
            
        st_features = to_st_features(features)
        long_term_feature_buffer = long_term_proposal_buffer = None
        if self.long_term:
            lt1_st_features = to_st_features(lt1_features)
            lt2_st_features = to_st_features(lt2_features)

            long_term_feature_buffer, long_term_proposal_buffer = self.roi_heads.preprocess_lt_support_frames(
                [lt1_st_features, lt2_st_features],
                [lt1_tubelet_proposals, lt2_tubelet_proposals],
                [lt1_gt_instances, lt2_gt_instances],
                self.train_reference_frame_idx
            )

        _, detector_losses = self.roi_heads(images, st_features, tubelet_proposals, self.train_reference_frame_idx, gt_instances, long_term_feature_buffer, long_term_proposal_buffer)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            same as in :meth:`forward`.
        """
        assert not self.training

        assert len(batched_inputs[0])!=1, "batch size > 1"
        if self.current_video and self.current_video != batched_inputs[0]['video']:  # New video
            if self.long_term:
                self.long_term_feature_buffer.clear()
                self.long_term_roi_buffer.clear()
            self.short_term_feature_buffer.clear()
            self.predict_proposals = None
            self.predict_objectness_logits = None
            self.frame_idx = 0
            print('New video.')  # New video
        

        self.current_video = batched_inputs[0]['video']
            
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        self.short_term_feature_buffer.append(features)

        if detected_instances is None:
            if self.proposal_generator:
                proposals, _, self.predict_proposals, self.predict_objectness_logits = self.proposal_generator(
                    images, features, self.reference_frame_idx, self.predict_proposals, self.predict_objectness_logits, None
                )

            else:
                raise NotImplementedError("Not implemented error.")
                # assert "proposals" in batched_inputs[0]
                # poposals = [x["proposals"].to(self.device) for x in batched_inputs]
            
            if len(self.short_term_feature_buffer) != self.num_frames:
                # we need to process more frames before we can give a detection set
                return None

            results, features_cur, long_term_curr_proposals, self.long_term_roi_buffer = self.roi_heads(
                images,
                self.short_term_feature_buffer,
                proposals,
                self.reference_frame_idx,
                None,
                self.long_term_feature_buffer,
                self.long_term_roi_buffer
            )
            if self.long_term and ((self.frame_idx % self.keyframe_interval) == 0):
                self.long_term_feature_buffer.append(features_cur)
                self.long_term_roi_buffer.append(long_term_curr_proposals)

            self.frame_idx += 1

        else:
            raise NotImplementedError("Not implemented error.")
            # detected_instances = [x.to(self.device) for x in detected_instances]
            # results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            return GeneralizedST_RCNN._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    @staticmethod
    def _postprocess(instances, batched_inputs, image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results
