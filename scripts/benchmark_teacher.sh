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
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-5}
#export CUDA_VISIBLE_DEVICES=5
source $PYTHON_ENV/conda_lbc/bin/activate
conda activate py35_10
export PYTHONPATH="$CODE/BEVSEG/lbc/PythonAPI:$PYTHONPATH"

##################################################
##### Instructions for General Researchers
##################################################

NAME=original_teacher
NAME=teacher_3
#NAME=teacher_noshift_0
NAME=e2e_bevseg_vpn_0
NAME=e2e_bevseg_vpn_2
NAME=pretrained_vpn_e2e_unfrozentrain_3
#NAME=pretrained_vpn_e2e_3
#NAME=teacher_noaug_4
NAME=pretrained_vpn_e2e_3
NAME=pretrained_vpn_frozen_7
NAME=pretrained_vpn_unfrozen_8

python benchmark_agent.py \
--model-path=./ckpts/$NAME/model-32.th \
--max-run 25 \
--port 3000 \
--suite=empty \
--show \
#--suite=dense \
#--model-path=./ckpts/$NAME/model-32.th \
#--model-path=./ckpts/$NAME/model-64.th \
#--port 2000 \

<< sample_cmd
cksbatch --nodelist=freddie ~/BEVSEG/lbc/scripts/benchmark_teacher.sh
bash ~/BEVSEG/lbc/scripts/benchmark_teacher.sh
sample_cmd
