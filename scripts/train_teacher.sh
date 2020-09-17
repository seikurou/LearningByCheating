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
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}
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
NAME=teacher_3
NAME=e2e_bevseg_vpn_2

mkdir -p ../ckpts/$NAME
#python train_birdview.py --dataset_dir=../data/original_data --log_dir=../ckpts/$NAME \
python train_birdview.py --dataset_dir=../data/1280res_150ksamples --log_dir=../ckpts/$NAME \
--bev_net vpn \
--batch_size 20 \
--num_workers 5 \
--angle_jitter 0 \
--x_jitter 0 \
--y_jitter 0 \



<< sample_cmd
bash ~/BEVSEG/lbc/scripts/train_teacher.sh
cksbatch --nodelist=freddie ~/BEVSEG/lbc/scripts/train_teacher.sh
sample_cmd
