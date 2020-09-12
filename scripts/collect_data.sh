#!/usr/bin/env bash

##################################################
##### Additional Instructions for Riselab Researchers
##################################################

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 2-00:00
#SBATCH --cpus-per-task 10
cd $CODE/BEVSEG/LearningByCheating

##################################################
##### Instructions for General Researchers
##################################################

# in one of the tab, run the simulator
DISPLAY= simulator/CarlaUE4.sh -opengl -fps=10 -benchmark  # display off
#CUDA_VISIBLE_DEVICES=1 ./CarlaUE4.sh -fps=10 -benchmark  # this is untested

export PYTHONPATH="$CODE/BEVSEG/LearningByCheating/PythonAPI:$PYTHONPATH"
mkdir -p ./data
python data_collector.py --dataset_path=data


<< sample_cmd
bash ~/BEVSEG/LearningByCheating/scripts/setup_on_rise_machiens.sh
sample_cmd
