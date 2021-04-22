import copy
import logging
import os
import numpy as np
import torch
from fvcore.common.file_io import PathManager
from PIL import Image

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

from .detection_utils import read_image, annotations_to_instances, build_transform_gen

"""
This file contains the default mapping that's applied to "dataset dicts".
"""

__all__ = ["DatasetMapper"]


class DatasetMapper:
    """
    A callable which takes a dataset dict in a
    spatio-temporal version of the Detectron2 Dataset format,
    and map it into a format used by the model.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    def __init__(self, cfg, is_train=True):
        self.tfm_gens = build_transform_gen(cfg, is_train)

        self.img_format     = cfg.INPUT.FORMAT
        self.mask_on        = cfg.MODEL.MASK_ON
        self.mask_format    = cfg.INPUT.MASK_FORMAT
        self.load_proposals = cfg.MODEL.LOAD_PROPOSALS

        if self.load_proposals:
            self.min_box_side_len = cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE
            self.proposal_topk = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
        self.is_train = is_train

    def __call__(self, dataset_dicts):
        """
        Args:
            dataset_dicts (list[dict]): Metadata of one frame batch,
            each frame in Detectron2 Dataset format.

        Returns:
            list[dict]: batch of N input frames each item in 
            a format that builtin models in detectron2 accept
        """
        output_dicts = []

        transforms = None
        dataset_dicts = copy.deepcopy(dataset_dicts)  # it will be modified by code below
        for frame_dict in dataset_dicts:
            image = read_image(frame_dict["file_name"], format=self.img_format)
            utils.check_image_size(frame_dict, image)

            # First, generate the TransformList for the first image (It has random components!).
            # Then, apply the same transformations fo the next images.
            # This way, we are applying the same transformation to the whole batch.
            if transforms is None:
                image, transforms = T.apply_transform_gens(self.tfm_gens, image)
            else:
                image = transforms.apply_image(image)

            image_shape = image.shape[:2]  # h, w

            # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
            # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
            # Therefore it's important to use torch.Tensor.
            frame_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

            # USER: Remove if you don't use pre-computed proposals.
            if self.load_proposals:
                utils.transform_proposals(
                    frame_dict, image_shape, transforms, self.min_box_side_len, self.proposal_topk
                )

            if not self.is_train:
                # USER: Modify this if you want to keep them for some reason.
                frame_dict.pop("annotations", None)
                frame_dict.pop("sem_seg_file_name", None)
                return frame_dict

            if "annotations" in frame_dict:
                # USER: Modify this if you want to keep them for some reason.
                for anno in frame_dict["annotations"]:
                        anno.pop("segmentation", None)
                        anno.pop("keypoints", None)

                # USER: Implement additional transformations if you have other types of data
                annos = [
                    utils.transform_instance_annotations(
                        obj, transforms, image_shape
                    )
                    for obj in frame_dict.pop("annotations")
                    if obj.get("iscrowd", 0) == 0
                ]
                instances = annotations_to_instances(
                    annos, image_shape
                )
                
                frame_dict["instances"] = utils.filter_empty_instances(instances)

            output_dicts.append(frame_dict)
        return output_dicts