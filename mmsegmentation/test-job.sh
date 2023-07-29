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

# 활성화된 환경에서 코드 실행.s

# BeiT 할 때 test CFG 꼭 수정해줘야함!
# PORT=29503 sh tools/dist_test.sh _satellite/upernet_convnext-b_ver19_fold0/convnext-base_upernet_8xb2-ade20k_fold0.py _satellite/upernet_convnext-b_ver19_fold0/best_mDice_iter_108000.pth 4 --work-dir _satellite/upernet_convnext-b_ver19_fold0-test
# PORT=29504 sh tools/dist_test.sh _satellite/upernet_convnext-b_ver19_fold1/convnext-base_upernet_8xb2-ade20k_fold1.py _satellite/upernet_convnext-b_ver19_fold1/best_mDice_iter_112000.pth 4 --work-dir _satellite/upernet_convnext-b_ver19_fold1-test
# PORT=29505 sh tools/dist_test.sh _satellite/upernet_convnext-b_ver19_fold2/convnext-base_upernet_8xb2-ade20k_fold2.py _satellite/upernet_convnext-b_ver19_fold2/best_mDice_iter_128000.pth 4 --work-dir _satellite/upernet_convnext-b_ver19_fold2-test
PORT=29506 sh tools/dist_test.sh _satellite/upernet_convnext-b_ver19_fold3/convnext-base_upernet_8xb2-ade20k_fold3.py _satellite/upernet_convnext-b_ver19_fold3/best_mDice_iter_100000.pth 3 --work-dir _satellite/upernet_convnext-b_ver19_fold3-test
# python inference.py 0         

echo "###"
echo "### END DATE=$(date)"
EOF


# sbatch --gres=gpu:4 --time=30:00 test-job.sh