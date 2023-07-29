# Mask2Former
Mask2Former: Masked-attention Mask Transformer for Universal Image Segmentation (CVPR 2022) 기반으로 작성되었습니다.

## 설치

설치 방법

## Pretrained Weight
coco instance segmentation pretrained model에서부터 학습을 시작합니다.
```
mkdir checkpoints
cd checkpoints
wget https://shi-labs.com/projects/dinat/checkpoints/m2f/mask2former_dinat_large_coco_instance.pth
cd ..
```

## 학습

- fold0 Training
```
python train_net.py \
--config-file configs/dacon/semantic-segmentation/mfs_f0_base448_dinatl_augver7.yaml \
--num-gpus {num_gpus} \
OUTPUT_DIR output/fold0
```
- fold1 Training
```
python train_net.py \
--config-file configs/dacon/semantic-segmentation/mfs_f1_base448_dinatl_augver7.yaml \
--num-gpus {num_gpus} \
OUTPUT_DIR output/fold1
```
- fold2 Training
```
python train_net.py \
--config-file configs/dacon/semantic-segmentation/mfs_f2_base448_dinatl_augver7.yaml \
--num-gpus {num_gpus} \
OUTPUT_DIR output/fold2
```
- fold3 Training
```
python train_net.py \
--config-file configs/dacon/semantic-segmentation/mfs_f3_base448_dinatl_augver7.yaml \
--num-gpus {num_gpus} \
OUTPUT_DIR output/fold3
```
- fold4 Training
```
python train_net.py \
--config-file configs/dacon/semantic-segmentation/mfs_f4_base448_dinatl_augver7.yaml \
--num-gpus {num_gpus} \
OUTPUT_DIR output/fold4
```


## Model ZOO

| 폴드   | 가중치 및 링크                               |
|-------|-------------------------------------------|
| fold0 | [weight(link)](https://example.com/fold0) |
| fold1 | [weight(link)](https://example.com/fold1) |
| fold2 | [weight(link)](https://example.com/fold2) |
| fold3 | [weight(link)](https://example.com/fold3) |
| fold4 | [weight(link)](https://example.com/fold4) |

## 추론

```

```




## License

Shield: [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

The majority of Mask2Former is licensed under a [MIT License](LICENSE).


However portions of the project are available under separate license terms: Swin-Transformer-Semantic-Segmentation is licensed under the [MIT license](https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation/blob/main/LICENSE), Deformable-DETR is licensed under the [Apache-2.0 License](https://github.com/fundamentalvision/Deformable-DETR/blob/main/LICENSE).

## <a name="CitingMask2Former"></a>Citing Mask2Former

If you use Mask2Former in your research or wish to refer to the baseline results published in the [Model Zoo](MODEL_ZOO.md), please use the following BibTeX entry.

```BibTeX
@inproceedings{cheng2021mask2former,
  title={Masked-attention Mask Transformer for Universal Image Segmentation},
  author={Bowen Cheng and Ishan Misra and Alexander G. Schwing and Alexander Kirillov and Rohit Girdhar},
  journal={CVPR},
  year={2022}
}
```

If you find the code useful, please also consider the following BibTeX entry.

```BibTeX
@inproceedings{cheng2021maskformer,
  title={Per-Pixel Classification is Not All You Need for Semantic Segmentation},
  author={Bowen Cheng and Alexander G. Schwing and Alexander Kirillov},
  journal={NeurIPS},
  year={2021}
}
```

## Acknowledgement

Code is largely based on MaskFormer (https://github.com/facebookresearch/MaskFormer).
