# Mask2Former
Mask2Former: Masked-attention Mask Transformer for Universal Image Segmentation (CVPR 2022) 기반으로 작성되었습니다.

## 설치

- Docker 이용하는 방법 (권장)
  - 모든 환경이 설치된 docker 이미지를 docker hub에 올려놨습니다.
  - 이미지 : `youkind/d2:latest`
  - 이미지 pull : `docker pull youkind/d2:latest`
  - 도커 생성 예시 (gpu 4개)
  ```
  docker run -it --gpus='"device=0,1,2,3"' --ipc=host -e NVIDIA_VISIBLE_DEVICES=all -v ./:/workspace --name spaceai youkind/d2:latest bash
  ```
  혹은 (CUDA 버전에 따라 다릅니다)
  ```
  docker run -it --runtime=nvidia --ipc=host -e NVIDIA_VISIBLE_DEVICES=all -v .:/workspace --name spaceai youkind/d2:latest bash
  ```
  와 같이 도커를 생성할 수 있고, 그 도커 안에서 아래 코드만 실행해주시면 detectron2 기반 training/inference가 가능합니다.
  ```
  cd mask2former/modeling/pixel_decoder/ops
  sh make.sh
  ```


- 직접 환경 설치하는 방법 (권장하지 않음)
  - 사용된 라이브러리 버전은 `libray.txt`에 기록되어 있으나, 다시 한번 말씀드리지만 제가 모든 환경을 세팅해놓은 도커로 실행하시는 것이 안전합니다.
  - mask2former는 detectron2 위에서 동작하기 때문에 detectron2를 설치하셔야 합니다.
  ```
  # detectron2
  git clone https://github.com/facebookresearch/detectron2.git
  cd detectron2
  pip install -v -e .
  ```
  - 이후 mask2former의 `deformableattn`을 컴파일해야 하는데, 아래와 같이 실행해주시면 됩니다.
  ```
  cd Dacon_SpaceAI_Solution # 최상위폴더에서 진행
  cd Mask2former/mask2former/modeling/pixel_decoder/ops
  sh mask.sh
  ```

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

- 직접 트레이닝한 `.pth` 파일로 인퍼런스할 경우, 아래와 같은 방식으로 `inference.py`를 실행시켜주시면 됩니다.
- 
```
python inference.py \
--config configs/dacon/semantic-segmentation/mfs_f{폴드번호}_base448_dinatl_augver7.yaml \
--model {모델 파일 경로} \
--output ../results/dinat_mask2former_fold{폴드번호}
``` 

- 예를 들어, `output/fold0/model_final.pth`로 인퍼런스를 진행하고 싶은 경우, 아래와 같이 실행하면 됩니다.
```
python inference.py \
--config configs/dacon/semantic-segmentation/mfs_f0_base448_dinatl_augver7.yaml \
--model output/fold0/model_final.pth \
--output ../results/dinat_mask2former_fold0
``` 

- 인퍼런스한 결과는 `../results/dinat_mask2former_fold{폴드번호}.csv` 에서 확인하실 수 있습니다.

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
