#!/bin/bash

echo "### START DATE=$(date)"
echo "### HOSTNAME=$(hostname)"
echo "### CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# conda 환경 활성화.
source ~/.bashrc
conda activate mmseg2

# cuda 11.0 환경 구성.
ml purge
ml load cuda/11.2

# 활성화된 환경에서 코드 실행.
# sh tools/dist_train.sh configs/_minyong/convnext-xlarge_upernet_8xb2-ade20k_fold0.py 4 --work-dir _satellite/upernet_convnext-l_ver1_fold0 --amp
# PORT=29501 sh tools/dist_train.sh configs/_minyong/convnext-xlarge_upernet_8xb2-ade20k_fold1.py 4 --work-dir _satellite/upernet_convnext-l_ver1_fold1 --amp
# sh tools/dist_train.sh configs/_minyong/convnext-xlarge_upernet_8xb2-ade20k_fold2.py 4 --work-dir _satellite/upernet_convnext-l_ver1_fold2 --amp
# PORT=29501 sh tools/dist_train.sh configs/_minyong/convnext-xlarge_upernet_8xb2-ade20k_fold3.py 4 --work-dir _satellite/upernet_convnext-l_ver1_fold3 --amp
# sh tools/dist_train.sh configs/_minyong/convnext-xlarge_upernet_8xb2-ade20k_fold4.py 4 --work-dir _satellite/upernet_convnext-l_ver1_fold4 --amp


# sh tools/dist_train.sh configs/_minyong/convnext-base_upernet_8xb2-ade20k_fold0.py 4 --work-dir _satellite/upernet_convnext-b_ver19_fold0 --amp
# PORT=29501 sh tools/dist_train.sh configs/_minyong/convnext-base_upernet_8xb2-ade20k_fold1.py 4 --work-dir _satellite/upernet_convnext-b_ver19_fold1 --amp
# sh tools/dist_train.sh configs/_minyong/convnext-base_upernet_8xb2-ade20k_fold2.py 4 --work-dir _satellite/upernet_convnext-b_ver19_fold2 --amp
PORT=29501 sh tools/dist_train.sh configs/_minyong/convnext-base_upernet_8xb2-ade20k_fold3.py 4 --work-dir _satellite/upernet_convnext-b_ver19_fold3 --amp
# sh tools/dist_train.sh configs/_minyong/convnext-base_upernet_8xb2-ade20k_fold4.py 4 --work-dir _satellite/upernet_convnext-b_ver19_fold4 --amp


# sh tools/dist_train.sh configs/_minyong/beit-base_upernet_8xb2_fold0.py 4 --work-dir _satellite/beit-b_upernet_ver8_fold0 --amp
# PORT=29501 sh tools/dist_train.sh configs/_minyong/beit-base_upernet_8xb2_fold1.py 4 --work-dir _satellite/beit-b_upernet_ver8_fold1 --amp
# sh tools/dist_train.sh configs/_minyong/beit-base_upernet_8xb2_fold2.py 4 --work-dir _satellite/beit-b_upernet_ver8_fold2 --amp
# PORT=29501 sh tools/dist_train.sh configs/_minyong/beit-base_upernet_8xb2_fold3.py 4 --work-dir _satellite/beit-b_upernet_ver8_fold3 --amp
# sh tools/dist_train.sh configs/_minyong/beit-base_upernet_8xb2_fold4.py 4 --work-dir _satellite/beit-b_upernet_ver8_fold4 --amp


echo "###"
echo "### END DATE=$(date)"
EOF

# sbatch --gres=gpu:4 --time=10:00:00 train-job.sh
# sbatch -q big_qos --gres=gpu:4 --time=10:00:00 train-job.sh