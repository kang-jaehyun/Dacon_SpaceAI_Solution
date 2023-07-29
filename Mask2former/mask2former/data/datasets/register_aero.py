# Copyright (c) Facebook, Inc. and its affiliates.
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

AERO_SEM_SEG_CATEGORIES = [
    {"name": "building", "color": [230, 25, 75]},
    {"name": "parking_lot", "color": [60, 180, 75]},
    {"name": "road", "color": [255, 225, 25]},
    {"name": "roadside_tree", "color": [0, 130, 200]},
    {"name": "paddy", "color": [245, 130, 48]},
    {"name": "field", "color": [145, 30, 180]},
    {"name": "forest", "color": [70, 240, 240]},
    {"name": "empty", "color": [240, 50, 230]},
    {"name": "else", "color": [210, 245, 60]},
]


def _get_aero_meta():
    stuff_classes = [k['name'] for k in AERO_SEM_SEG_CATEGORIES]
    stuff_colors = [k['color'] for k in AERO_SEM_SEG_CATEGORIES]

    ret = {
        "stuff_classes": stuff_classes,
        "stuff_colors": stuff_colors,
    }
    return ret


def register_all_aero(root):
    root = os.path.join(root, "satelite")
    meta = _get_aero_meta()
    for name, dirname in [("train", "Training"), ("val", "Validation")]:
        image_dir = os.path.join(root, dirname, "images")
        gt_dir = os.path.join(root, dirname, "annotations")
        name = f"aero_sem_seg_{name}"
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="tif", image_ext="tif")
        )
        MetadataCatalog.get(name).set(
            stuff_classes=meta["stuff_classes"][:],
            stuff_colors=meta["stuff_colors"][:],
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=256, 
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_aero(_root)
