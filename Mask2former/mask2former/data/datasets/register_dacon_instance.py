# Copyright (c) Facebook, Inc. and its affiliates.
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
from detectron2.utils.file_io import PathManager
from detectron2.data.datasets.coco import load_coco_json, register_coco_instances

DACON_CATEGORIES = [
    {"id": 1, "name": "building"},
]

_PREDEFINED_SPLITS = {
    # point annotations without masks
    "dacon_instance_train": (
        "dacon/train_img",
        "dacon/train_annotations.json",
    ),
    "dacon_instance_slice_train": (
        "dacon/train_slice_img",
        "dacon/train_slice_annotations.json",
    ),
    "dacon_instance_slice_val": (
        "dacon/val_slice_img",
        "dacon/val_slice_annotations.json",
    ),
}

def _get_dacon_instances_meta():
    thing_ids = [k["id"] for k in DACON_CATEGORIES]
    # Mapping from the incontiguous ADE category id to an id in [0, 99]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in DACON_CATEGORIES]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
    }
    return ret

def register_all_dacon_instance(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_coco_instances(
            key,
            _get_dacon_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_dacon_instance(_root)