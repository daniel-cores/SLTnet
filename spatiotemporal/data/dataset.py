import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from .datasets.st_coco import load_coco_json

_DATA_DIR = "/datasets"

def register_vid_dataset(cfg):
    num_frames = cfg.MODEL.SPATIOTEMPORAL.NUM_FRAMES
    if cfg.MODEL.SPATIOTEMPORAL.FORWARD_AGGREGATION:
        # (f_{t-NUM_FRAMES}, ..., f_{t-1}, f_t, f_{t+1}, ..., f_{t+NUM_FRAMES})
        num_frames = (2 * num_frames) + 1

    splits = {
        "det_train_subsampled": (
            "ILSVRC2015/Data/DET",  # complete path with element id (in JSON). See _prep_roidb_entry
            "ILSVRC2015/annotations_pytorch/det_subsampled_{}f.json".format(num_frames)
        ),

        "vid_train_split0": (
            "vid/ILSVRC/Data/VID",  # complete path with element id (in JSON). See _prep_roidb_entry
            "vid/annotations_pytorch/vid_train_ILSVRC2015_VID_train_0000.json"
        ),
        "vid_train_split1": (
            "vid/ILSVRC/Data/VID",  # complete path with element id (in JSON). See _prep_roidb_entry
            "vid/annotations_pytorch/vid_train_ILSVRC2015_VID_train_0001.json"
        ),
        "vid_train_split2": (
            "vid/ILSVRC/Data/VID",  # complete path with element id (in JSON). See _prep_roidb_entry
            "vid/annotations_pytorch/vid_train_ILSVRC2015_VID_train_0002.json"
        ),
        "vid_train_split3": (
            "vid/ILSVRC/Data/VID",  # complete path with element id (in JSON). See _prep_roidb_entry
            "vid/annotations_pytorch/vid_train_ILSVRC2015_VID_train_0003.json"
        ),

        "vid_val": (
            "vid/ILSVRC/Data/VID",  # complete path with element id (in JSON). See _prep_roidb_entry
            "vid/annotations_pytorch/vid_val.json"
        )
    }

    SPATIOTEMPORAL_KEYS = ['id_track', 'video']
    for key, (image_root, json_file) in splits.items():
        json_file = os.path.join(_DATA_DIR, json_file)
        image_root = os.path.join(_DATA_DIR, image_root)

        DatasetCatalog.register(
            key,
            lambda key=key, json_file=json_file, image_root=image_root: load_coco_json(
                json_file, image_root, key, extra_annotation_keys=SPATIOTEMPORAL_KEYS
            ),
        )