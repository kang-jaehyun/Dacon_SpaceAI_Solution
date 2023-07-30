# mmsegmentation

## 환경 설치
- 이미 해당 과정을 완료했다면, 바로 모델 학습 및 추론 단계를 진행하면 됩니다.
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

## 모델 학습
- `Dacon_SpaceAI_Solution/mmsegmentation` directory에서 아래 과정을 진행합니다.
- mmsegmentation 라이브러리를 통한 학습에서 모델은 총 3가지를 활용하였으며, 각 모델 구조별로 5 Fold로 나눈 모든 데이터에 대해 학습을 진행합니다.
  - ConvNext + UperNet
  - ResNest + DeeplabV3+
  - Segformer

- 아래 명령어는 multi-GPU (4개)를 기준으로 구성되어 있습니다.
  - `sh tools/dist_train.sh {config_file} {num_gpus} --work-dir {work_directory} --amp`
- 아래 과정을 한번에 수행하기 위해서는 `bash train.sh` 명령어를 활용해주세요.

```
PORT=29500 sh tools/dist_train.sh configs/convnext-base_upernet_8xb2-ade20k_fold0.py 4 --work-dir _satellite/convnext-base_fold0 --amp
PORT=29501 sh tools/dist_train.sh configs/convnext-base_upernet_8xb2-ade20k_fold1.py 4 --work-dir _satellite/convnext-base_fold1 --amp
PORT=29502 sh tools/dist_train.sh configs/convnext-base_upernet_8xb2-ade20k_fold2.py 4 --work-dir _satellite/convnext-base_fold2 --amp
PORT=29503 sh tools/dist_train.sh configs/convnext-base_upernet_8xb2-ade20k_fold3.py 4 --work-dir _satellite/convnext-base_fold3 --amp
PORT=29504 sh tools/dist_train.sh configs/convnext-base_upernet_8xb2-ade20k_fold4.py 4 --work-dir _satellite/convnext-base_fold4 --amp


PORT=29500 sh tools/dist_train.sh configs/resnest_deeplabv3plus_fold0.py 4 --work-dir _satellite/resnest_deeplabv3plus_fold0 --amp
PORT=29501 sh tools/dist_train.sh configs/resnest_deeplabv3plus_fold1.py 4 --work-dir _satellite/resnest_deeplabv3plus_fold1 --amp
PORT=29502 sh tools/dist_train.sh configs/resnest_deeplabv3plus_fold2.py 4 --work-dir _satellite/resnest_deeplabv3plus_fold2 --amp
PORT=29503 sh tools/dist_train.sh configs/resnest_deeplabv3plus_fold3.py 4 --work-dir _satellite/resnest_deeplabv3plus_fold3 --amp
PORT=29504 sh tools/dist_train.sh configs/resnest_deeplabv3plus_fold4.py 4 --work-dir _satellite/resnest_deeplabv3plus_fold4 --amp


PORT=29500 sh tools/dist_train.sh configs/segformer_fold0.py 4 --work-dir _satellite/segformer_fold0 --amp
PORT=29501 sh tools/dist_train.sh configs/segformer_fold1.py 4 --work-dir _satellite/segformer_fold1 --amp
PORT=29502 sh tools/dist_train.sh configs/segformer_fold2.py 4 --work-dir _satellite/segformer_fold2 --amp
PORT=29503 sh tools/dist_train.sh configs/segformer_fold3.py 4 --work-dir _satellite/segformer_fold3 --amp
PORT=29504 sh tools/dist_train.sh configs/segformer_fold4.py 4 --work-dir _satellite/segformer_fold4 --amp
```


## 제공된 model weight 기반 추론
- `Dacon_SpaceAI_Solution/mmsegmentation` directory에서 아래 과정을 진행합니다.
- 학습을 통해 얻은 모델을 활용하여 test data에 대한 추론을 진행합니다.
- 상위 폴더의 `Dacon_SpaceAI_Solution/model_weights` directory에 있는 model weight 중 mmsegmentation으로 학습한 15개 모델에 대한 추론을 진행합니다.
  - 만약 다운로드 하지 않았다면 미리 제공된 링크를 통해 학습된 모델의 weight를 다운로드 받아주세요.
- 아래 명령어는 multi-gpu (4개)를 기준으로 구성되어 있습니다.
    - `sh tools/dist_test.sh {config_file} {model_weight} {num_gpus} --work-dir {work_directory}`를 사용하면 됩니다.
- 각 모델을 통해 추론한 mask 바탕으로 `rle_decoding.py` 파일을 통해 15개의 csv 파일을 만듭니다. 각각의 csv file은 최종 앙상블을 위해 모두 상위 폴더인 `Dacon_SpaceAI_Solution/results` directory에 저장됩니다.
- 아래 과정을 한번에 수행하기 위해서는 `bash inference.sh` 명령어를 활용해주세요.

```
PORT=29500 sh tools/dist_test.sh configs/convnext-base_upernet_8xb2-ade20k_fold0.py ../model_weights/convnext_upernet_fold0.pth 4 --work-dir _satellite/convnext-base_fold0
PORT=29501 sh tools/dist_test.sh configs/convnext-base_upernet_8xb2-ade20k_fold1.py ../model_weights/convnext_upernet_fold1.pth 4 --work-dir _satellite/convnext-base_fold1
PORT=29502 sh tools/dist_test.sh configs/convnext-base_upernet_8xb2-ade20k_fold2.py ../model_weights/convnext_upernet_fold2.pth 4 --work-dir _satellite/convnext-base_fold2
PORT=29503 sh tools/dist_test.sh configs/convnext-base_upernet_8xb2-ade20k_fold3.py ../model_weights/convnext_upernet_fold3.pth 4 --work-dir _satellite/convnext-base_fold3
PORT=29504 sh tools/dist_test.sh configs/convnext-base_upernet_8xb2-ade20k_fold4.py ../model_weights/convnext_upernet_fold4.pth 4 --work-dir _satellite/convnext-base_fold4


PORT=29500 sh tools/dist_test.sh configs/resnest_deeplabv3plus_fold0.py ../model_weights/resnest_deeplabv3plus_fold0.pth 4 --work-dir _satellite/resnest_deeplabv3plus_fold0
PORT=29501 sh tools/dist_test.sh configs/resnest_deeplabv3plus_fold1.py ../model_weights/resnest_deeplabv3plus_fold1.pth 4 --work-dir _satellite/resnest_deeplabv3plus_fold1
PORT=29502 sh tools/dist_test.sh configs/resnest_deeplabv3plus_fold2.py ../model_weights/resnest_deeplabv3plus_fold2.pth 4 --work-dir _satellite/resnest_deeplabv3plus_fold2
PORT=29503 sh tools/dist_test.sh configs/resnest_deeplabv3plus_fold3.py ../model_weights/resnest_deeplabv3plus_fold3.pth 4 --work-dir _satellite/resnest_deeplabv3plus_fold3
PORT=29504 sh tools/dist_test.sh configs/resnest_deeplabv3plus_fold4.py ../model_weights/resnest_deeplabv3plus_fold4.pth 4 --work-dir _satellite/resnest_deeplabv3plus_fold4


PORT=29500 sh tools/dist_test.sh configs/segformer_fold0.py ../model_weights/segformer_fold0.pth 4 --work-dir _satellite/segformer_fold0
PORT=29501 sh tools/dist_test.sh configs/segformer_fold1.py ../model_weights/segformer_fold1.pth 4 --work-dir _satellite/segformer_fold1
PORT=29502 sh tools/dist_test.sh configs/segformer_fold2.py ../model_weights/segformer_fold2.pth 4 --work-dir _satellite/segformer_fold2
PORT=29503 sh tools/dist_test.sh configs/segformer_fold3.py ../model_weights/segformer_fold3.pth 4 --work-dir _satellite/segformer_fold3
PORT=29504 sh tools/dist_test.sh configs/segformer_fold4.py ../model_weights/segformer_fold4.pth 4 --work-dir _satellite/segformer_fold4

python rle_encoding.py
```





## Acknowledgement

MMSegmentation is an open source project that welcome any contribution and feedback.
We wish that the toolbox and benchmark could serve the growing research
community by providing a flexible as well as standardized toolkit to reimplement existing methods
and develop their own new semantic segmentation methods.

## Citation

If you find this project useful in your research, please consider cite:

```bibtex
@misc{mmseg2020,
    title={{MMSegmentation}: OpenMMLab Semantic Segmentation Toolbox and Benchmark},
    author={MMSegmentation Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmsegmentation}},
    year={2020}
}
```