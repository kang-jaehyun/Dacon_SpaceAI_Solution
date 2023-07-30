#!/bin/bash

echo "### START DATE=$(date)"
echo "### HOSTNAME=$(hostname)"
echo "### CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# conda 환경 활성화.
source ~/.bashrc
conda activate mmseg3

# cuda 11.0 환경 구성.
ml purge
ml load cuda/11.2

# PORT=29500 sh tools/dist_test.sh configs/convnext-base_upernet_8xb2-ade20k_fold0.py ../model_weights/convnext_upernet_fold0.pth 4 --work-dir _satellite/convnext-base_fold0
# PORT=29501 sh tools/dist_test.sh configs/convnext-base_upernet_8xb2-ade20k_fold1.py ../model_weights/convnext_upernet_fold1.pth 4 --work-dir _satellite/convnext-base_fold1
# PORT=29502 sh tools/dist_test.sh configs/convnext-base_upernet_8xb2-ade20k_fold2.py ../model_weights/convnext_upernet_fold2.pth 4 --work-dir _satellite/convnext-base_fold2
# PORT=29503 sh tools/dist_test.sh configs/convnext-base_upernet_8xb2-ade20k_fold3.py ../model_weights/convnext_upernet_fold3.pth 4 --work-dir _satellite/convnext-base_fold3
# PORT=29504 sh tools/dist_test.sh configs/convnext-base_upernet_8xb2-ade20k_fold4.py ../model_weights/convnext_upernet_fold4.pth 4 --work-dir _satellite/convnext-base_fold4


PORT=29500 sh tools/dist_test.sh configs/resnest_deeplabv3plus_fold0.py ../model_weights/resnest_deeplabv3plus_fold0.pth 4 --work-dir _satellite/resnest_deeplabv3plus_fold0
PORT=29501 sh tools/dist_test.sh configs/resnest_deeplabv3plus_fold1.py ../model_weights/resnest_deeplabv3plus_fold1.pth 4 --work-dir _satellite/resnest_deeplabv3plus_fold1
PORT=29502 sh tools/dist_test.sh configs/resnest_deeplabv3plus_fold2.py ../model_weights/resnest_deeplabv3plus_fold2.pth 4 --work-dir _satellite/resnest_deeplabv3plus_fold2
PORT=29503 sh tools/dist_test.sh configs/resnest_deeplabv3plus_fold3.py ../model_weights/resnest_deeplabv3plus_fold3.pth 4 --work-dir _satellite/resnest_deeplabv3plus_fold3
PORT=29504 sh tools/dist_test.sh configs/resnest_deeplabv3plus_fold4.py ../model_weights/resnest_deeplabv3plus_fold4.pth 4 --work-dir _satellite/resnest_deeplabv3plus_fold4


# PORT=29500 sh tools/dist_test.sh configs/segformer_fold0.py ../model_weights/segformer_fold0.pth 4 --work-dir _satellite/segformer_fold0
# PORT=29501 sh tools/dist_test.sh configs/segformer_fold1.py ../model_weights/segformer_fold1.pth 4 --work-dir _satellite/segformer_fold1
# PORT=29502 sh tools/dist_test.sh configs/segformer_fold2.py ../model_weights/segformer_fold2.pth 4 --work-dir _satellite/segformer_fold2
# PORT=29503 sh tools/dist_test.sh configs/segformer_fold3.py ../model_weights/segformer_fold3.pth 4 --work-dir _satellite/segformer_fold3
# PORT=29504 sh tools/dist_test.sh configs/segformer_fold4.py ../model_weights/segformer_fold4.pth 4 --work-dir _satellite/segformer_fold4


# python rle_encoding.py


echo "###"
echo "### END DATE=$(date)"
EOF


# sbatch --gres=gpu:4 --time=30:00 test-job.sh