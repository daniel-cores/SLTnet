import bisect
import copy
import itertools
import logging
import numpy as np
import operator
import pickle
import torch.utils.data
import os
from fvcore.common.file_io import PathManager
from tabulate import tabulate
from termcolor import colored

from detectron2.structures import BoxMode
from detectron2.utils.comm import get_world_size
from detectron2.utils.env import seed_all_rng
from detectron2.utils.logger import log_first_n


from detectron2.data import samplers
from detectron2.data.catalog import DatasetCatalog, MetadataCatalog
from detectron2.data.common import AspectRatioGroupedDataset, DatasetFromList, MapDataset
from detectron2.data.detection_utils import check_metadata_consistency

from spatiotemporal.data.dataset_mapper import DatasetMapper

"""
This file contains the default logic to build a dataloader for training or testing working with video.

Based on:
https://github.com/facebookresearch/detectron2/blob/master/detectron2/data/build.py
"""

__all__ = [
    "build_detection_train_loader",
    "build_detection_test_loader",
    "get_detection_dataset_dicts",
    "load_proposals_into_dataset",
    "print_instances_class_histogram",
]


def filter_invalid_frame_groups(grouped_dataset_dicts, num_frames):
    """
    Filter out frame groups with incomplete gt tubelets
    Args:
        dataset_dicts (list[list[dict]]): annotations in Detectron2 Dataset format (already batched).
    Returns:
        list[list[dict]]: the same format, but filtered.
    """
    num_before = sum([len(v) for v in grouped_dataset_dicts])
    def valid(group):
        # Check for complete GT tubes
        gt_ids = [[] for _ in group]
        for idx, f in enumerate(group):
            # Remove crowd annotations in train
            f["annotations"] = [ann for ann in f["annotations"] if ann.get("iscrowd", 0) == 0]
            for ann in f["annotations"]:
                # if ann.get("iscrowd", 0) != 0:
                #     return False
                gt_ids[idx].append(ann['id_track'])

        return gt_ids.count(gt_ids[0]) == len(gt_ids) and len(gt_ids[0]) > 0

    grouped_dataset_dicts = [
        [(b_id, b) for b_id, b in video_batches if valid(b)] for video_batches in grouped_dataset_dicts
    ]
    num_after = sum([len(v) for v in grouped_dataset_dicts])
    logger = logging.getLogger(__name__)
    logger.info(
        "Removed {} groups with no usable annotations. {} groups left.".format(
            num_before - num_after, num_after
        )
    )
    return grouped_dataset_dicts


def load_proposals_into_dataset(dataset_dicts, proposal_file):
    """
    Load precomputed object proposals into the dataset.
    The proposal file should be a pickled dict with the following keys:
    - "ids": list[int] or list[str], the image ids
    - "boxes": list[np.ndarray], each is an Nx4 array of boxes corresponding to the image id
    - "objectness_logits": list[np.ndarray], each is an N sized array of objectness scores
        corresponding to the boxes.
    - "bbox_mode": the BoxMode of the boxes array. Defaults to ``BoxMode.XYXY_ABS``.
    Args:
        dataset_dicts (list[dict]): annotations in Detectron2 Dataset format.
        proposal_file (str): file path of pre-computed proposals, in pkl format.
    Returns:
        list[dict]: the same format as dataset_dicts, but added proposal field.
    """
    logger = logging.getLogger(__name__)
    logger.info("Loading proposals from: {}".format(proposal_file))

    with PathManager.open(proposal_file, "rb") as f:
        proposals = pickle.load(f, encoding="latin1")

    # Rename the key names in D1 proposal files
    rename_keys = {"indexes": "ids", "scores": "objectness_logits"}
    for key in rename_keys:
        if key in proposals:
            proposals[rename_keys[key]] = proposals.pop(key)

    # Fetch the indexes of all proposals that are in the dataset
    # Convert image_id to str since they could be int.
    img_ids = set({str(record["image_id"]) for record in dataset_dicts})
    id_to_index = {str(id): i for i, id in enumerate(proposals["ids"]) if str(id) in img_ids}

    # Assuming default bbox_mode of precomputed proposals are 'XYXY_ABS'
    bbox_mode = BoxMode(proposals["bbox_mode"]) if "bbox_mode" in proposals else BoxMode.XYXY_ABS

    for record in dataset_dicts:
        # Get the index of the proposal
        i = id_to_index[str(record["image_id"])]

        boxes = proposals["boxes"][i]
        objectness_logits = proposals["objectness_logits"][i]
        # Sort the proposals in descending order of the scores
        inds = objectness_logits.argsort()[::-1]
        record["proposal_boxes"] = boxes[inds]
        record["proposal_objectness_logits"] = objectness_logits[inds]
        record["proposal_bbox_mode"] = bbox_mode

    return dataset_dicts


def print_instances_class_histogram(dataset_dicts, class_names):
    """
    Args:
        dataset_dicts (list[dict]): list of dataset dicts.
        class_names (list[str]): list of class names (zero-indexed).
    """
    num_classes = len(class_names)
    hist_bins = np.arange(num_classes + 1)
    histogram = np.zeros((num_classes,), dtype=np.int)
    for entry in dataset_dicts:
        annos = entry["annotations"]
        classes = [x["category_id"] for x in annos if not x.get("iscrowd", 0)]
        histogram += np.histogram(classes, bins=hist_bins)[0]

    N_COLS = min(6, len(class_names) * 2)

    def short_name(x):
        # make long class names shorter. useful for lvis
        if len(x) > 13:
            return x[:11] + ".."
        return x

    data = list(
        itertools.chain(*[[short_name(class_names[i]), int(v)] for i, v in enumerate(histogram)])
    )
    total_num_instances = sum(data[1::2])
    data.extend([None] * (N_COLS - (len(data) % N_COLS)))
    if num_classes > 1:
        data.extend(["total", total_num_instances])
    data = itertools.zip_longest(*[data[i::N_COLS] for i in range(N_COLS)])
    table = tabulate(
        data,
        headers=["category", "#instances"] * (N_COLS // 2),
        tablefmt="pipe",
        numalign="left",
        stralign="center",
    )
    log_first_n(
        logging.INFO,
        "Distribution of instances among all {} categories:\n".format(num_classes)
        + colored(table, "cyan"),
        key="message",
    )


def get_detection_dataset_dicts(
    cfg, dataset_names, frames_per_group, train=True, proposal_files=None, long_term=False
):
    """
    Load and prepare dataset dicts for instance detection/segmentation and semantic segmentation.
    Args:
        dataset_names (list[str]): a list of dataset names
        filter_empty (bool): whether to filter out images without instance annotations
        proposal_files (list[str]): if given, a list of object proposal files
            that match each dataset in `dataset_names`.
    """

    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    assert len(dataset_names)
    dataset_dicts = [DatasetCatalog.get(dataset_name) for dataset_name in dataset_names]
    for dataset_name, dicts in zip(dataset_names, dataset_dicts):
        assert len(dicts), "Dataset '{}' is empty!".format(dataset_name)

    if proposal_files is not None:
        assert len(dataset_names) == len(proposal_files)
        # load precomputed proposals from proposal files
        dataset_dicts = [
            load_proposals_into_dataset(dataset_i_dicts, proposal_file)
            for dataset_i_dicts, proposal_file in zip(dataset_dicts, proposal_files)
        ]

    dataset_dicts = list(itertools.chain.from_iterable(dataset_dicts))
    
    # Group frames by video
    dataset_dicts = itertools.groupby(
        dataset_dicts, key=lambda x:x["video"]
    )
    dataset_dicts = [list(v) for k, v in dataset_dicts]
        
    if train:
        # Build frame batches per video
        dataset_dicts = [list(chunks(video_dicts, frames_per_group)) for video_dicts in dataset_dicts]

        # Assign consecutive batch ids per video
        dataset_dicts = [[(b_id, batch) for b_id, batch in enumerate(video_batches)] for video_batches in dataset_dicts]
        
        # Filter batches by length (last batch of each video might be shorter than frames_per_group)
        dataset_dicts = [
            [(b_id, batch) for b_id, batch in video_batches if len(batch) == frames_per_group]
            for video_batches in dataset_dicts
        ]

        # Remove batches with incomplete GT tubelets
        dataset_dicts = filter_invalid_frame_groups(dataset_dicts, frames_per_group)

        # Subsample videos
        if cfg.DATALOADER.TRAIN_SUBSAMPLING:
            for video_idx in range(len(dataset_dicts)):

                if long_term:
                    video_copy = copy.deepcopy(dataset_dicts[video_idx])
                    # Add long-term support frames
                    for i in range(len(video_copy)):

                        if len(video_copy) > 1:
                            valid = list(range(len(video_copy)))
                            valid.remove(i)
                            batch_1 = np.random.choice(valid)
                            if len(video_copy) > 2: # We can't select two different long-term batches
                                valid.remove(batch_1)
                            batch_2 = np.random.choice(valid)
                        else:
                            batch_1 = 0
                            batch_2 = 0

                        dataset_dicts[video_idx][i] = \
                            (video_copy[i][0], video_copy[i][1] + video_copy[batch_1][1] + video_copy[batch_2][1])

                if len(dataset_dicts[video_idx]) <=  cfg.DATALOADER.TRAIN_GROUPS_PER_VIDEO * frames_per_group:
                    continue

                filtered_batch_ids = [b_id for b_id, _ in dataset_dicts[video_idx]]

                # Select TRAIN_GROUPS_PER_VIDEO batches evenly spaced from the first valid batch to the last one
                sampled_idx = np.round(
                    np.linspace(filtered_batch_ids[0], filtered_batch_ids[-1], cfg.DATALOADER.TRAIN_GROUPS_PER_VIDEO)
                )

                # Map sampled_idx to filtered dataset_dicts (removing short batches and incomplete GT tubelets)
                mapped_idx = np.searchsorted(filtered_batch_ids, sampled_idx)

                # Apply the mask
                dataset_dicts[video_idx] = [dataset_dicts[video_idx][idx] for idx in mapped_idx]
        
        # Remove batch ids
        dataset_dicts = [[batch for _, batch in video_batches] for video_batches in dataset_dicts]
    
        dataset_dicts = list(itertools.chain.from_iterable(dataset_dicts))
    
    else:  # test: batch size = 1
        # frame padding at the beginning of each video
        frame_padding = cfg.MODEL.SPATIOTEMPORAL.NUM_FRAMES - 1

        for v_idx in range(len(dataset_dicts)):
            v = dataset_dicts[v_idx]
            for frame in v:
                frame['is_padding'] = False

            padding = []
            for _ in range(frame_padding):
                padding.append(copy.deepcopy(v[0]))
                padding[-1]['is_padding'] = True

            dataset_dicts[v_idx] = padding + v

        
        # frame padding at the end of each video
        if cfg.MODEL.SPATIOTEMPORAL.FORWARD_AGGREGATION:
            for v_idx in range(len(dataset_dicts)):
                v = dataset_dicts[v_idx]
                padding = []
                for _ in range(frame_padding+1):
                    padding.append(copy.deepcopy(v[-1]))
                    padding[-1]['is_padding'] = True

                # We need one extra left frame padding 
                dataset_dicts[v_idx] =  [copy.deepcopy(v[0])] + v + padding
                dataset_dicts[v_idx][0]['is_padding'] = True

        dataset_dicts = list(itertools.chain.from_iterable(dataset_dicts))
        dataset_dicts = [[dataset_dict] for dataset_dict in dataset_dicts]
    
    logger = logging.getLogger(__name__)
    logger.info(
        "Generating {} frame groups with {} frames.".format(
            len(dataset_dicts), frames_per_group
        )
    )

    return dataset_dicts


def build_detection_train_loader(cfg, mapper=None):
    """
    A data loader is created by the following steps:
    1. Use the dataset names in config to query :class:`DatasetCatalog`, and obtain a list of dicts.
    2. Start workers to work on the dicts. Each worker will:
      * Map each metadata dict into another format to be consumed by the model.
      * Batch them by simply putting dicts into a list.
    The batched ``list[mapped_dict]`` is what this dataloader will return.
    Args:
        cfg (CfgNode): the config
        mapper (callable): a callable which takes a sample (dict) from dataset and
            returns the format to be consumed by the model.
            By default it will be `DatasetMapper(cfg, True)`.
    Returns:
        an infinite iterator of training data
    """
    # Change the batching strategy to replicate the N-1 first frames of each video
    # to not load frames from different videos in the same batch

    num_workers = get_world_size()
    images_per_batch = cfg.SOLVER.IMS_PER_BATCH
    
    num_frames = cfg.MODEL.SPATIOTEMPORAL.NUM_FRAMES
    if cfg.MODEL.SPATIOTEMPORAL.FORWARD_AGGREGATION:
        # (f_{t-NUM_FRAMES}, ..., f_{t-1}, f_t, f_{t+1}, ..., f_{t+NUM_FRAMES})
        num_frames = (2 * num_frames) + 1

    assert (
        images_per_batch == 1
    ), "SOLVER.IMS_PER_BATCH ({}) must be 1. Actual batch size in spatio-temporal dataset must be set to num_frames({}).".format(
        images_per_batch, num_frames
    )
    images_per_batch = num_frames

    assert (
        images_per_batch % num_workers == 0
    ), "SOLVER.IMS_PER_BATCH ({}) must be divisible by the number of workers ({}).".format(
        images_per_batch, num_workers
    )
    assert (
        images_per_batch >= num_workers
    ), "SOLVER.IMS_PER_BATCH ({}) must be larger than the number of workers ({}).".format(
        images_per_batch, num_workers
    )
    images_per_worker = images_per_batch // num_workers

    dataset_dicts = get_detection_dataset_dicts(
        cfg,
        cfg.DATASETS.TRAIN,
        num_frames,
        proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
        long_term=cfg.MODEL.SPATIOTEMPORAL.LONG_TERM
    )
    dataset = DatasetFromList(dataset_dicts, copy=False)

    if mapper is None:
        mapper = DatasetMapper(cfg, True)
    dataset = MapDataset(dataset, mapper)

    sampler = samplers.TrainingSampler(len(dataset), shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_sampler=None,
        collate_fn=operator.itemgetter(0),  # don't batch, but yield individual elements
        worker_init_fn=worker_init_reset_seed,
    )  # yield individual mapped dict
    
    return data_loader


def build_detection_test_loader(cfg, dataset_name, mapper=None):
    """
    Similar to `build_detection_train_loader`.
    But this function uses the given `dataset_name` argument (instead of the names in cfg),
    and uses batch size 1.
    Args:
        cfg: a detectron2 CfgNode
        dataset_name (str): a name of the dataset that's available in the DatasetCatalog
        mapper (callable): a callable which takes a sample (dict) from dataset
           and returns the format to be consumed by the model.
           By default it will be `DatasetMapper(cfg, False)`.
    Returns:
        DataLoader: a torch DataLoader, that loads the given detection
        dataset, with test-time transformation and batching.
    """
    num_frames = cfg.MODEL.SPATIOTEMPORAL.NUM_FRAMES
    if cfg.MODEL.SPATIOTEMPORAL.FORWARD_AGGREGATION:
        # (f_{t-NUM_FRAMES}, ..., f_{t-1}, f_t, f_{t+1}, ..., f_{t+NUM_FRAMES})
        num_frames = (2 * num_frames) + 1

    dataset_dicts = get_detection_dataset_dicts(
        cfg,
        [dataset_name],
        num_frames,
        train=False,
        proposal_files=[
            cfg.DATASETS.PROPOSAL_FILES_TEST[list(cfg.DATASETS.TEST).index(dataset_name)]
        ]
        if cfg.MODEL.LOAD_PROPOSALS
        else None,
    )

    dataset = DatasetFromList(dataset_dicts)
    if mapper is None:
        mapper = DatasetMapper(cfg, False)
    dataset = MapDataset(dataset, mapper)

    sampler = samplers.InferenceSampler(len(dataset))
    # Always use 1 image per worker during inference since this is the
    # standard when reporting inference time in papers.
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, 1, drop_last=False)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
    )
    return data_loader


def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    return batch


def worker_init_reset_seed(worker_id):
    seed_all_rng(np.random.randint(2 ** 31) + worker_id)