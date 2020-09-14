#!/usr/bin/env bash

##################################################
##### Additional Instructions for Riselab Researchers
##################################################

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 2-00:00
#SBATCH --cpus-per-task 10
cd $CODE/BEVSEG/lbc

REPO=BEVSEG/lbc
mkdir -p $CODE/$REPO $DATA/$REPO

############################## installing conda
cd $PYTHON_ENV
wget https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh
bash Anaconda3-2020.07-Linux-x86_64.sh
# when prompted, install it at $PYTHON_ENV/conda_lbc

# create a python 3.5 environment, 3.5 required for their egg file installation
source $PYTHON_ENV/conda_lbc/bin/activate
conda create -n py35_10 python=3.5.5
conda activate py35_10

ln -s $DATA/data/lbc $CODE/BEVSEG/lbc/data
ln -s $DATA/data/carla_simulator $CODE/BEVSEG/lbc/simulator
mkdir -p $DATA/BEVSEG/lbc/ckpts
ln -s $DATA/BEVSEG/lbc/ckpts $CODE/BEVSEG/lbc/ckpts

cd $CODE/BEVSEG/lbc
##################################################
##### Instructions for General Researchers
##################################################

############################## install necessary packages
# There are mixtures of pip and conda because this is only a hack to make it work :)
pip install -r $CODE/knowledge/requirements.txt  # this is my custom packages I always install, ask me for it haha
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

############################## setup the carla simulator
mkdir -p ./simulator
cd simulator
wget http://carla-assets-internal.s3.amazonaws.com/Releases/Linux/CARLA_0.9.6.tar.gz
tar -xvzf CARLA_0.9.6.tar.gz

<< sample_cmd
bash ~/BEVSEG/lbc/scripts/setup_on_rise_machiens.sh
sample_cmd
