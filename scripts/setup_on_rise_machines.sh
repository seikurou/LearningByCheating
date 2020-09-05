#!/usr/bin/env bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 2-00:00
#SBATCH --cpus-per-task 10

# instructions for setting up the lbc environment on the rise machines
ROOT=/data/ck/carla_lbc/carla_lbc/
CONDA=/data/ck/conda/p38_5

############################## installing conda
cd $ROOT
wget https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh
bash "$ROOT/Anaconda3-2020.07-Linux-x86_64.sh"
# install it at $CONDA, then the code from below will continue to work

# create a python 3.5 environment, 3.5 required for their egg file installation
source $CONDA/bin/activate
conda create -n py35_10 python=3.5.5
conda activate py35_10

############################## install necessary packages
# There are mixtures of pip and conda because this is only a hack to make lbc work on rise
pip install -r ~/knowledge/requirements.txt  # this is my custom packages I always install, ask me for it haha
pip install torch==1.0.0 torchvision==0.2.1 pygame pillow==6.0
pip install lmdb
pip install opencv-python
pip install pygame tqdm

# install their custom carla package
cd PythonAPI/carla/dist
rm carla-0.9.6-py3.5-linux-x86_64.egg
wget http://www.cs.utexas.edu/~dchen/lbc_release/egg/carla-0.9.6-py3.5-linux-x86_64.egg
python -m easy_install carla-0.9.6-py3.5-linux-x86_64.egg

# instsall even more packages, doing it in this order because we just want something that works
conda  install -c conda-forge jpeg  # this somehow fixes the jpeg error
pip install loguru==0.3.0

############################## sample data collection cmd

# in one of the tab, run the simulator
cd $ROOT
DISPLAY= ./CarlaUE4.sh -opengl -fps=10 -benchmark  # display off
#CUDA_VISIBLE_DEVICES=1 ./CarlaUE4.sh -fps=10 -benchmark  # this is untested

cd $ROOT
export PYTHONPATH="$ROOT/PythonAPI:$PYTHONPATH"
python data_collector.py --dataset_path=$ROOT/data



<< sample_cmd
bash ~/BEVSEG/LearningByCheating/scripts/setup_on_rise_machiens.sh
sample_cmd
