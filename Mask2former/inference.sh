#!/bin/bash
echo "### START DATE=$(date)"
echo "### HOSTNAME=$(hostname)"
echo "### CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo $SLURM_NODELIST
echo $SLURM_NODEID

# conda 환경 활성화.
source ~/.bashrc


# cuda 11.3 환경 구성.
# ml purge
# ml load cuda/11.0
# ml av | grep cuda
# module spider cuda/11.3
# ml swap  cuda/11.7   nccl/2.8.4/cuda11.2
# which nvcc ; echo ; nvcc -V

docker=" /home/jaehyunkang/d2.simg"
exec_file=" python /home/jaehyunkang/Mask2former-MP/inference_mp_tta_aidc.py"
fold=1

ml purge
ml load singularity
# singularity exec --nv $docker $exec_file $config_file $machine_cfg $out_dir $dataloader_cpu
singularity exec --nv $docker $exec_file \
--config /home/jaehyunkang/Mask2former-MP/configs/dacon/semantic-segmentation/mfs_f${fold}_base448_dinatl_augver7.yaml \
--model /home/jaehyunkang/tensorboard/final/mfs_f${fold}_base448_dinatl_augver7/model_final.pth \
--tta 0 \
--num_process 6 \
--output /home/jaehyunkang/Mask2former-MP/submissions/mfs_f${fold}_base448_dinatl_augver7_final-notta.csv

echo "###"
echo "### END DATE=$(date)"
EOF

# sbatch --gres=gpu:4 --time=72:00:00 inference.sh
# sbatch -q big_qos --gres=gpu:8 --nodelist=node02 --time=72:00:00 inference.sh
# sbatch -p big -q big_qos --gres=gpu:7 --nodelist=node13 --time=72:00:00 inference.sh

# sbatch --gres=gpu:6 --time=72:00:00 train_job.sh
# sbatch -q big_qos --gres=gpu:6 --time=72:00:00 train_job.sh
# sbatch -p big -q big_qos --gres=gpu:6 --time=72:00:00 train_job.sh