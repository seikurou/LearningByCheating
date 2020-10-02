#!/usr/bin/env bash

##################################################
##### Additional Instructions for Riselab Researchers
##################################################

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 2-00:00
#SBATCH --cpus-per-task 10
#SBATCH  --gres gpu:1

cd $CODE/BEVSEG/lbc
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}
source $PYTHON_ENV/conda_lbc/bin/activate
conda activate py35_10
export PYTHONPATH=$CODE/network_library/:$PYTHONPATH
export PYTHONPATH="$CODE/BEVSEG/lbc/PythonAPI:$PYTHONPATH"

##################################################
##### Instructions for General Researchers
##################################################

cd training
NAME=teacher_noshift_0
NAME=teacher_norot_0
NAME=teacher_noaug_0
NAME=debug
NAME=e2e_bevseg_vpn_2
NAME=pretrained_vpn_e2e_3
#NAME=pretrained_vpn_e2e_unfrozentrain_3
#NAME=teacher_noangle_5
#NAME=teacher_noaug_5
#NAME=teacher_noshift_6
#NAME=teacher_7
NAME=pretrained_vpn_frozen_7
NAME=pretrained_vpn_unfrozen_8
NAME=vpn_noise_0

mkdir -p ../ckpts/$NAME
#python train_birdview.py --dataset_dir=../data/original_data --log_dir=../ckpts/$NAME \
python train_birdview.py --log_dir=../ckpts/$NAME \
--batch_size 60 \
--num_workers 8 \
--x_jitter 0 \
--y_jitter 0 \
--angle_jitter 0 \
--bev_net vpn \
--bev_channel 2 \
--bev_freeze \
--dataset_dir=../data/1280res_150ksamples_noise \
#--dataset_dir=../data/1280res_150ksamples \



<< sample_cmd
bash ~/BEVSEG/lbc/scripts/train_teacher.sh
cksbatch --nodelist=freddie ~/BEVSEG/lbc/scripts/train_teacher.sh
rsync_local_data_to_remote_data /data/ck/BESEG/baseline_vpn/runs/lbc_noise_7/ flaminio freddie
sample_cmd
