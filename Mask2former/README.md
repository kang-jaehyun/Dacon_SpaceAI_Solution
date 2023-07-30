# Mask2Former
Mask2Former: Masked-attention Mask Transformer for Universal Image Segmentation (CVPR 2022) 기반으로 작성되었습니다.

## 설치

- Docker 이용하는 방법 (권장)
  - 모든 환경이 설치된 docker 이미지를 docker hub에 올려놨습니다.
  - 이미지 : `youkind/d2:latest`
  - 이미지 pull : `docker pull youkind/d2:latest`

- 직접 환경 설치하는 방법 (권장하지 않음)
  - mask2former는 detectron2 위에서 동작하기 때문에 detectron2를 설치하셔야 합니다.
  - 

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
- 직접 트레이닝을 하지 않고 결과를 확인하기 위해서는 학습된 weight를 받아야 합니다.
- [weight 다운로드는 여기에서 받으실 수 있습니다.](https://yonsei-my.sharepoint.com/personal/youkind_o365_yonsei_ac_kr/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fyoukind%5Fo365%5Fyonsei%5Fac%5Fkr%2FDocuments%2Fspaceai%2Fmodel%5Fweights&view=0)
- 다운로드가 완료된 `.pth` 파일들은 `Dacon_SpaceAI_Solution/model_weights`안에 넣어주세요.
  
## 추론 (Inference)

- 모델 `.pth` 파일을 `dinat_mask2former_fold{0~4}.pth`로 저장하셨을 경우, 아래 코드를 통해 5개 fold의 인퍼런스를 진행하실 수 있습니다.

```
sh inference.sh
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
