# SW중심대학 공동 AI 경진대회 2023

## SpaceAI

- 강재현
- 강민용
- 박선종

## 폴더 구조

```
- datasets/                          # preprocessed dataset
  - Satelite/                        # WILL BE CREATED after running data_preprocessing.sh
  - refined_train.parquet
  - make_refind_dataset.py
  - ...
- EDA                                # EDA
- original_data/                     # 데이콘 제공 기본 데이터 저장 폴더
- Mask2former/                       # detectron2 src
- mmseg/                             # mmsegmentation src
- results                            # model outputs
```

## 환경세팅

- 대회 기간 2개의 라이브러리 (mmsegmentation, detectron2)를 활용해서 학습을 진행했으며, 결과 재현을 위해서 2가지 다른 환경에서의 학습 및 추론이 필요합니다.
- <b>데이터 전처리 / mmsegmentation 학습 및 추론 / 최종 앙상블</b>을 위해서 아래 명령어로 구성된 conda 환경 설정을 진행해주세요.

```
conda create -n mmseg python=3.9
conda activate mmseg

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install -U openmim
mim install "mmengine==0.8.2"
mim install "mmcv==2.0.1"
pip install "mmsegmentation==1.1.0"
pip install "mmpretrain==1.0.0"
pip install "cityscapesScripts==2.2.2"
pip install pyarrow
```

- <b>Detectron2 학습 및 추론</b>을 위한 환경 세팅은 아래의 모델 학습 및 추론 부분을 참고해주세요.

## 데이터 전처리

- 데이터 전처리를 위해서 [original_data] folder에 제공된 데이터셋의 train_img / test_img를

- 균형있는 모델 학습 및 검증을 위해 전처리를 진행하며, 아래와 같은 과정을 통해 진행됩니다
  1. 라벨링이 잘못되어 있는 학습 이미지 제거 (직접 검수한 106개 학습 이미지 삭제)
  2. 이미지에 분포되어 있는 building segment (4-way connection)의 개수를 기준으로 Stratified 5-Fold로 데이터셋 분리
  3. validation data에 대해 (1024, 1024) 이미지 -> 16 \* (256, 256) 이미지로 나눔
- 잘못 라벨링된 이미지 삭제(106개) 및 학습 데이터를 5-Fold로 나눈 parquet file을 통해서 실제 학습에 사용할 데이터를 아래 명령어를 통해 만들 수 있습니다.
  - `bash data_preprocessing.sh`
  - test data는 제공된 데이터 그대로 datasets에 복사합니다.
  - parquet file은 다음과 같은 column으로 구성되어 있습니다, `['img_id', 'img_path', 'mask_rle', 'fold']`
  - `EDA/building_segment_number.ipynb` 을 통해서 5-Fold를 나눈 기준 (number of building segment in 4-way connection) 및 분포를 직접 확인할 수 있습니다.

## 사용 모델 종류

| Model                 | Framework      | weights (fold0)                                                                                   | weights (fold1)                                                                                   | weights (fold2)                                                                                   | weights (fold3)                                                                                   | weights (fold4)                                                                                   |
| --------------------- | -------------- | ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| DiNAT-L + Mask2former | detectron2     | [download](https://drive.google.com/file/d/1zeN6QBn8WiBm5Rg4S5U_VkF9ysPJW9lx/view?usp=drive_link) | [download](https://drive.google.com/file/d/1I-0P4oFAn5YGymkTHiIYbS6h0ijrbyTU/view?usp=drive_link) | [download](https://drive.google.com/file/d/1O-Lrv-HqGuWgqgBd3jNEFSKFDH3XAyc0/view?usp=drive_link) | [download](https://drive.google.com/file/d/1iH3czOEOlJN5KXCO_MI-1heXFFwYyYAV/view?usp=drive_link) | [download](https://drive.google.com/file/d/1bhm45n5X17UkO3LH5yK01O8J96RpvpV8/view?usp=drive_link) |
| ConvNext + Upernet    | mmsegmentation | [download](https://drive.google.com/file/d/1sKOdjgYCJs3O04AKBYOqlNsffRrvRuCs/view?usp=drive_link) | [download](https://drive.google.com/file/d/1o41_VkupyoP1BZzK3810tfrgJPObOZFR/view?usp=drive_link) | [download](https://drive.google.com/file/d/1dnPU8vWqz2g4xioEThlditSexw4EBBJl/view?usp=drive_link) | [download](https://drive.google.com/file/d/1--IQzq62_jEfOnf-EIan6S4O6bEzRkqJ/view?usp=drive_link) | [download](https://drive.google.com/file/d/1MO61u1FL-B-GdgV0Yes38zIEIGbetUMB/view?usp=drive_link) |
| ResNest + DeeplabV3+  | mmsegmentation | [download](https://drive.google.com/file/d/1v3BEabo3nM-rJ_YJL7ZOOKEiU74Pn3D-/view?usp=drive_link) | [download](https://drive.google.com/file/d/1aaEglP3pKUwx0mofe17Q1OJIdbIvjzsm/view?usp=drive_link) | [download](https://drive.google.com/file/d/1RIhXyGxo8BxqN64UKD7YtT-SaREDUxcJ/view?usp=drive_link) | [download](https://drive.google.com/file/d/15nDjvdy8oNK8Is5mftedwGckB5y7mX4H/view?usp=drive_link) | [download](https://drive.google.com/file/d/1Y-rQOiXh8YCyvHsVY-iDaje1gV7JJFEU/view?usp=drive_link) |
| Segformer             | mmsegentation  | [download](https://drive.google.com/file/d/1tyO-2A_Ge2dVsPFWbOCl6XDrLyfGUxWf/view?usp=drive_link) | [download](https://drive.google.com/file/d/12AFTXESz2HMLgFndlz67LfIzudqrTnSp/view?usp=drive_link) | [download](https://drive.google.com/file/d/1mrH-_158oHUht3rVcM8ahVT2wpoXZSBJ/view?usp=drive_link) | [download](https://drive.google.com/file/d/1N4636OTiGtSpheiUcjy29uy3GcB9Pmkb/view?usp=drive_link) | [download](https://drive.google.com/file/d/1ihGyYFBpyCyD1GsH1e0f35tqkEUvHTHr/view?usp=drive_link) |

- 전체 weight 다운로드는 [여기](https://drive.google.com/drive/folders/1paQhzjF7JcsEbCMr1z1bV2rj766M-nhg?usp=drive_link)에서 받으실 수 있습니다.
- 다운로드가 완료된 `.pth` 파일들은 `Dacon_SpaceAI_Solution/model_weights`안에 넣어주세요.
- 해당 사용된 pretrained weighted 들은 모두 각각의 configs에 공개된 형태로 COCO Dataset 또는 ADE20k Dataset으로 구성되었으며 학습 및 추론을 시킬시 공개된 링크 주소를 통해 접근 가능합니다.
- ConvNext: ImageNet21k
- ResNest : ImageNet21k
- Segformer : ADE-20k
- DiNAT-L : COCO

## 모델 학습 및 추론

- 각각의 라이브러리를 활용한 모델의 학습 및 추론 과정이 필요합니다.
- 제공된 pretrained weight를 통해 추론을 완료하면, `Dacon_SpaceAI_Solution/results` directory에 모든 모델(20개)의 결과가 모입니다.

### Mask2former

detectron2 기반 Mask2former 결과 재현을 위해서는 [Mask2former 학습 및 추론](Mask2former/README.md)을 참고해주세요.

### mmseg

mmsegmentation 기반 모델들의 결과 재현을 위해서는 [mmsegmentation 학습 및 추론](mmsegmentation/README.md)을 참고해주세요.

## 최종 앙상블

최종 앙상블은 각 폴드에 대해서 5번(fold0 ~ fold4), 그리고 모든 폴드에 대해서 마지막으로 한번 이루어집니다. <br>
이를 위해서는 `results` 폴더 안에 아래와 같은 파일들이 준비되어야 합니다.

```
results
- ensemble.py
- convnext-base_fold0.csv
- ...
- convnext-base_fold4.csv
- segformer_fold0.csv
- ...
- segformer_fold4.csv
- resnest_deeplabv3plus_fold0.csv
- ...
- resnest_deeplabv3plus_fold4.csv
- dinat_mask2former_fold0.csv
- ...
- dinat_mask2former_fold4.csv
```

위와 같이 파일이 준비되었다면, `bash ensemble.sh`실행을 통해 앙상블 결과를 얻으실 수 있고, `results/ensemble.csv`에 결과가 저장됩니다.
