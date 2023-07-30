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
  3. validation data에 대해 (1024, 1024) 이미지 -> 16 * (256, 256) 이미지로 나눔
   
- 잘못 라벨링된 이미지 삭제(106개) 및 학습 데이터를 5-Fold로 나눈 parquet file을 통해서 실제 학습에 사용할 데이터를 아래 명령어를 통해 만들 수 있습니다.
  - `bash data_preprocessing.sh`
  - test data는 제공된 데이터 그대로 datasets에 복사합니다.
  - parquet file은 다음과 같은 column으로 구성되어 있습니다, `['img_id', 'img_path', 'mask_rle', 'fold']`
  - `EDA/segment_number.ipynb` 을 통해서 5-Fold를 나눈 기준 (number of building segment in 4-way connection) 및 분포를 직접 확인할 수 있습니다.

## 사용 모델 종류
| Model                 | Framework   | weights (fold0)                         | weights (fold1)                         | weights (fold2)                         | weights (fold3)                         | weights (fold4)                         |
|-----------------------|-------------|-----------------------------------------|-----------------------------------------|-----------------------------------------|-----------------------------------------|-----------------------------------------|
| DiNAT-L + Mask2former | detectron2  | [download](https://yonsei-my.sharepoint.com/personal/youkind_o365_yonsei_ac_kr/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fyoukind%5Fo365%5Fyonsei%5Fac%5Fkr%2FDocuments%2Fspaceai%2Fmodel%5Fweights%2Fdinat%5Fmask2former%5Ffold0%2Epth&parent=%2Fpersonal%2Fyoukind%5Fo365%5Fyonsei%5Fac%5Fkr%2FDocuments%2Fspaceai%2Fmodel%5Fweights) | [download](https://yonsei-my.sharepoint.com/personal/youkind_o365_yonsei_ac_kr/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fyoukind%5Fo365%5Fyonsei%5Fac%5Fkr%2FDocuments%2Fspaceai%2Fmodel%5Fweights%2Fdinat%5Fmask2former%5Ffold1%2Epth&parent=%2Fpersonal%2Fyoukind%5Fo365%5Fyonsei%5Fac%5Fkr%2FDocuments%2Fspaceai%2Fmodel%5Fweights) | [download](https://yonsei-my.sharepoint.com/personal/youkind_o365_yonsei_ac_kr/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fyoukind%5Fo365%5Fyonsei%5Fac%5Fkr%2FDocuments%2Fspaceai%2Fmodel%5Fweights%2Fdinat%5Fmask2former%5Ffold2%2Epth&parent=%2Fpersonal%2Fyoukind%5Fo365%5Fyonsei%5Fac%5Fkr%2FDocuments%2Fspaceai%2Fmodel%5Fweights) | [download](https://yonsei-my.sharepoint.com/personal/youkind_o365_yonsei_ac_kr/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fyoukind%5Fo365%5Fyonsei%5Fac%5Fkr%2FDocuments%2Fspaceai%2Fmodel%5Fweights%2Fdinat%5Fmask2former%5Ffold3%2Epth&parent=%2Fpersonal%2Fyoukind%5Fo365%5Fyonsei%5Fac%5Fkr%2FDocuments%2Fspaceai%2Fmodel%5Fweights) | [download](https://yonsei-my.sharepoint.com/personal/youkind_o365_yonsei_ac_kr/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fyoukind%5Fo365%5Fyonsei%5Fac%5Fkr%2FDocuments%2Fspaceai%2Fmodel%5Fweights%2Fdinat%5Fmask2former%5Ffold4%2Epth&parent=%2Fpersonal%2Fyoukind%5Fo365%5Fyonsei%5Fac%5Fkr%2FDocuments%2Fspaceai%2Fmodel%5Fweights) |
| ConvNext + Upernet    | mmsegmentation       | [download](https://yonsei-my.sharepoint.com/personal/youkind_o365_yonsei_ac_kr/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fyoukind%5Fo365%5Fyonsei%5Fac%5Fkr%2FDocuments%2Fspaceai%2Fmodel%5Fweights%2Fconvnext%5Fupernet%5Ffold0%2Epth&parent=%2Fpersonal%2Fyoukind%5Fo365%5Fyonsei%5Fac%5Fkr%2FDocuments%2Fspaceai%2Fmodel%5Fweights) | [download](https://yonsei-my.sharepoint.com/personal/youkind_o365_yonsei_ac_kr/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fyoukind%5Fo365%5Fyonsei%5Fac%5Fkr%2FDocuments%2Fspaceai%2Fmodel%5Fweights%2Fconvnext%5Fupernet%5Ffold1%2Epth&parent=%2Fpersonal%2Fyoukind%5Fo365%5Fyonsei%5Fac%5Fkr%2FDocuments%2Fspaceai%2Fmodel%5Fweights) | [download](https://yonsei-my.sharepoint.com/personal/youkind_o365_yonsei_ac_kr/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fyoukind%5Fo365%5Fyonsei%5Fac%5Fkr%2FDocuments%2Fspaceai%2Fmodel%5Fweights%2Fconvnext%5Fupernet%5Ffold2%2Epth&parent=%2Fpersonal%2Fyoukind%5Fo365%5Fyonsei%5Fac%5Fkr%2FDocuments%2Fspaceai%2Fmodel%5Fweights) | [download](https://yonsei-my.sharepoint.com/personal/youkind_o365_yonsei_ac_kr/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fyoukind%5Fo365%5Fyonsei%5Fac%5Fkr%2FDocuments%2Fspaceai%2Fmodel%5Fweights%2Fconvnext%5Fupernet%5Ffold3%2Epth&parent=%2Fpersonal%2Fyoukind%5Fo365%5Fyonsei%5Fac%5Fkr%2FDocuments%2Fspaceai%2Fmodel%5Fweights) | [download](https://yonsei-my.sharepoint.com/personal/youkind_o365_yonsei_ac_kr/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fyoukind%5Fo365%5Fyonsei%5Fac%5Fkr%2FDocuments%2Fspaceai%2Fmodel%5Fweights%2Fconvnext%5Fupernet%5Ffold4%2Epth&parent=%2Fpersonal%2Fyoukind%5Fo365%5Fyonsei%5Fac%5Fkr%2FDocuments%2Fspaceai%2Fmodel%5Fweights) |
| ResNest + DeeplabV3+  | mmsegmentation       | [download](https://yonsei-my.sharepoint.com/personal/youkind_o365_yonsei_ac_kr/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fyoukind%5Fo365%5Fyonsei%5Fac%5Fkr%2FDocuments%2Fspaceai%2Fmodel%5Fweights%2Fresnest%5Fdeeplabv3plus%5Ffold0%2Epth&parent=%2Fpersonal%2Fyoukind%5Fo365%5Fyonsei%5Fac%5Fkr%2FDocuments%2Fspaceai%2Fmodel%5Fweights) | [download](https://yonsei-my.sharepoint.com/personal/youkind_o365_yonsei_ac_kr/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fyoukind%5Fo365%5Fyonsei%5Fac%5Fkr%2FDocuments%2Fspaceai%2Fmodel%5Fweights%2Fresnest%5Fdeeplabv3plus%5Ffold1%2Epth&parent=%2Fpersonal%2Fyoukind%5Fo365%5Fyonsei%5Fac%5Fkr%2FDocuments%2Fspaceai%2Fmodel%5Fweights) | [download](https://yonsei-my.sharepoint.com/personal/youkind_o365_yonsei_ac_kr/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fyoukind%5Fo365%5Fyonsei%5Fac%5Fkr%2FDocuments%2Fspaceai%2Fmodel%5Fweights%2Fresnest%5Fdeeplabv3plus%5Ffold2%2Epth&parent=%2Fpersonal%2Fyoukind%5Fo365%5Fyonsei%5Fac%5Fkr%2FDocuments%2Fspaceai%2Fmodel%5Fweights) | [download](https://yonsei-my.sharepoint.com/personal/youkind_o365_yonsei_ac_kr/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fyoukind%5Fo365%5Fyonsei%5Fac%5Fkr%2FDocuments%2Fspaceai%2Fmodel%5Fweights%2Fresnest%5Fdeeplabv3plus%5Ffold3%2Epth&parent=%2Fpersonal%2Fyoukind%5Fo365%5Fyonsei%5Fac%5Fkr%2FDocuments%2Fspaceai%2Fmodel%5Fweights) | [download](https://yonsei-my.sharepoint.com/personal/youkind_o365_yonsei_ac_kr/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fyoukind%5Fo365%5Fyonsei%5Fac%5Fkr%2FDocuments%2Fspaceai%2Fmodel%5Fweights%2Fresnest%5Fdeeplabv3plus%5Ffold4%2Epth&parent=%2Fpersonal%2Fyoukind%5Fo365%5Fyonsei%5Fac%5Fkr%2FDocuments%2Fspaceai%2Fmodel%5Fweights) |
| Segformer             | mmsegentation       | [download](https://yonsei-my.sharepoint.com/personal/youkind_o365_yonsei_ac_kr/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fyoukind%5Fo365%5Fyonsei%5Fac%5Fkr%2FDocuments%2Fspaceai%2Fmodel%5Fweights%2Fsegformer%5Ffold0%2Epth&parent=%2Fpersonal%2Fyoukind%5Fo365%5Fyonsei%5Fac%5Fkr%2FDocuments%2Fspaceai%2Fmodel%5Fweights) | [download](https://yonsei-my.sharepoint.com/personal/youkind_o365_yonsei_ac_kr/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fyoukind%5Fo365%5Fyonsei%5Fac%5Fkr%2FDocuments%2Fspaceai%2Fmodel%5Fweights%2Fsegformer%5Ffold1%2Epth&parent=%2Fpersonal%2Fyoukind%5Fo365%5Fyonsei%5Fac%5Fkr%2FDocuments%2Fspaceai%2Fmodel%5Fweights) | [download](https://yonsei-my.sharepoint.com/personal/youkind_o365_yonsei_ac_kr/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fyoukind%5Fo365%5Fyonsei%5Fac%5Fkr%2FDocuments%2Fspaceai%2Fmodel%5Fweights%2Fsegformer%5Ffold2%2Epth&parent=%2Fpersonal%2Fyoukind%5Fo365%5Fyonsei%5Fac%5Fkr%2FDocuments%2Fspaceai%2Fmodel%5Fweights) | [download](https://yonsei-my.sharepoint.com/personal/youkind_o365_yonsei_ac_kr/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fyoukind%5Fo365%5Fyonsei%5Fac%5Fkr%2FDocuments%2Fspaceai%2Fmodel%5Fweights%2Fsegformer%5Ffold3%2Epth&parent=%2Fpersonal%2Fyoukind%5Fo365%5Fyonsei%5Fac%5Fkr%2FDocuments%2Fspaceai%2Fmodel%5Fweights) | [download](https://yonsei-my.sharepoint.com/personal/youkind_o365_yonsei_ac_kr/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fyoukind%5Fo365%5Fyonsei%5Fac%5Fkr%2FDocuments%2Fspaceai%2Fmodel%5Fweights%2Fsegformer%5Ffold4%2Epth&parent=%2Fpersonal%2Fyoukind%5Fo365%5Fyonsei%5Fac%5Fkr%2FDocuments%2Fspaceai%2Fmodel%5Fweights) |

- [weight 다운로드는 여기에서 받으실 수 있습니다.](https://yonsei-my.sharepoint.com/personal/youkind_o365_yonsei_ac_kr/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fyoukind%5Fo365%5Fyonsei%5Fac%5Fkr%2FDocuments%2Fspaceai%2Fmodel%5Fweights&view=0)
- 다운로드가 완료된 `.pth` 파일들은 `Dacon_SpaceAI_Solution/model_weights`안에 넣어주세요.

## 모델 학습 및 추론
- 각각의 라이브러리를 활용한 모델의 학습 및 추론 과정이 필요합니다.
- 제공된 pretrained weight를 통해 추론을 완료하면, `Dacon_SpaceAI_Solution/results` directory에 모든 모델(20개)의 결과가 모입니다.

### Mask2former
detectron2 기반 Mask2former 결과 재현을 위해서는 [Mask2former 학습 및 추론](Mask2former/README.md)을 참고해주세요.

### mmseg
mmsegmentation 기반 모델들의 결과 재현을 위해서는 [mmsegmentation 학습 및 추론](mmseg/README.md)을 참고해주세요.

## 최종 앙상블
TBW