# Copyright (c) Facebook, Inc. and its affiliates.
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
from detectron2.utils.file_io import PathManager

DACON_SEM_SEG_CATEGORIES = [
    {"name": "background", "color": [100, 100, 100]},
    {"name": "building", "color": [230, 25, 75]},
]


def _get_dacon_meta():
    stuff_classes = [k['name'] for k in DACON_SEM_SEG_CATEGORIES]
    stuff_colors = [k['color'] for k in DACON_SEM_SEG_CATEGORIES]

    ret = {
        "stuff_classes": stuff_classes,
        "stuff_colors": stuff_colors,
    }
    return ret


def load_sem_seg_test(image_root, image_ext="jpg"):
    """
    Load semantic segmentation datasets. All files under "gt_root" with "gt_ext" extension are
    treated as ground truth annotations and all files under "image_root" with "image_ext" extension
    as input images. Ground truth and input images are matched using file paths relative to
    "gt_root" and "image_root" respectively without taking into account file extensions.
    This works for COCO as well as some other datasets.

    Args:
        gt_root (str): full path to ground truth semantic segmentation files. Semantic segmentation
            annotations are stored as images with integer values in pixels that represent
            corresponding semantic labels.
        image_root (str): the directory where the input images are.
        gt_ext (str): file extension for ground truth annotations.
        image_ext (str): file extension for input images.

    Returns:
        list[dict]:
            a list of dicts in detectron2 standard format without instance-level
            annotation.

    Notes:
        1. This function does not read the image and ground truth files.
           The results do not have the "image" and "sem_seg" fields.
    """

    # We match input images with ground truth based on their relative filepaths (without file
    # extensions) starting from 'image_root' and 'gt_root' respectively.
    def file2id(folder_path, file_path):
        # extract relative path starting from `folder_path`
        image_id = os.path.normpath(os.path.relpath(file_path, start=folder_path))
        # remove file extension
        image_id = os.path.splitext(image_id)[0]
        return image_id

    input_files = sorted(
        (os.path.join(image_root, f) for f in PathManager.ls(image_root) if f.endswith(image_ext)),
        key=lambda file_path: file2id(image_root, file_path),
    )

    # Use the intersection, so that val2017_100 annotations can run smoothly with val2017 images

    dataset_dicts = []
    for (img_path) in input_files:
        record = {}
        record["file_name"] = img_path
        dataset_dicts.append(record)

    return dataset_dicts

def register_all_dacon(root):
    root = os.path.join(root, "dacon")
    meta = _get_dacon_meta()
    for name in ("train", "val", "train_slice", "val_slice", "train_fold", "train_fold_slice", "val_fold", "val_fold_slice"):
        image_dir = os.path.join(root, name+'_img')
        gt_dir = os.path.join(root, name+'_gt')
        name = f"dacon_sem_seg_{name}"
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="png")
        )
        MetadataCatalog.get(name).set(
            stuff_classes=meta["stuff_classes"][:],
            stuff_colors=meta["stuff_colors"][:],
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=256, 
        )
    DatasetCatalog.register(
            "dacon_sem_seg_test", lambda x="/workspace/Mask2former-MP/datasets/dacon/test_img": load_sem_seg_test(x, image_ext="png")
        )
    MetadataCatalog.get("dacon_sem_seg_test").set(
            stuff_classes=meta["stuff_classes"][:],
            stuff_colors=meta["stuff_colors"][:],
            image_root=image_dir,
            evaluator_type="sem_seg",
            ignore_label=256, 
        )
    for name in ('train', 'val_slice'):
        for i in range(5):
            image_dir = os.path.join(*[root, 'img_dir', f'{name}_{i}'])
            gt_dir = os.path.join(*[root, 'ann_dir', f'{name}_{i}'])
            foldname = f"dacon_sem_seg_{name}_fold{i}"
            DatasetCatalog.register(
                foldname, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="png")
            )
            MetadataCatalog.get(foldname).set(
                stuff_classes=meta["stuff_classes"][:],
                stuff_colors=meta["stuff_colors"][:],
                image_root=image_dir,
                sem_seg_root=gt_dir,
                evaluator_type="sem_seg",
                ignore_label=256, 
            )
_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_dacon(_root)
