# SW중심대학 공동 AI 경진대회 2023


## SpaceAI

대충 팀 소개

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

## 데이터 전처리
TBW

## 사용 모델 종류
| Model                 | Framework   |
|-----------------------|-------------|
| DiNAT-L + Mask2former | detectron2  |
| ConvNext + Upernet    | mmseg       |
| ResNest + DeeplabV3+  | mmseg       |
| Segformer             | mmseg       |

## 모델 학습 및 추론

### Mask2former
detectron2 기반 Mask2former 결과 재현을 위해서는 [Mask2former 학습 및 추론](Mask2former/README.md)을 참고해주세요.

### mmseg
mmsegmentation 기반 모델들의 결과 재현을 위해서는 [mmsegmentation 학습 및 추론](mmseg/README.md)을 참고해주세요.

## 최종 앙상블
TBW