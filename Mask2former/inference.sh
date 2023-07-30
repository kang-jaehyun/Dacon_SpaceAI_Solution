python inference.py \
--config configs/dacon/semantic-segmentation/mfs_f0_base448_dinatl_augver7.yaml \
--model ../model_weights/dinat_mask2former_fold0.pth

python inference.py \
--config configs/dacon/semantic-segmentation/mfs_f1_base448_dinatl_augver7.yaml \
--model ../model_weights/dinat_mask2former_fold1.pth

python inference.py \
--config configs/dacon/semantic-segmentation/mfs_f2_base448_dinatl_augver7.yaml \
--model ../model_weights/dinat_mask2former_fold2.pth

python inference.py \
--config configs/dacon/semantic-segmentation/mfs_f3_base448_dinatl_augver7.yaml \
--model ../model_weights/dinat_mask2former_fold3.pth

python inference.py \
--config configs/dacon/semantic-segmentation/mfs_f4_base448_dinatl_augver7.yaml \
--model ../model_weights/dinat_mask2former_fold4.pth

