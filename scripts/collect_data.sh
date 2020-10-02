#!/usr/bin/env bash

##################################################
##### Additional Instructions for Riselab Researchers
##################################################

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 2-00:00
#SBATCH --cpus-per-task 10
cd $CODE/BEVSEG/lbc
source $PYTHON_ENV/conda_lbc/bin/activate
conda activate py35_10
export PYTHONPATH="$CODE/BEVSEG/lbc/PythonAPI:$PYTHONPATH"

##################################################
##### Instructions for General Researchers
##################################################

# in one of the tab, run the simulator
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-5}
#DISPLAY= simulator/CarlaUE4.sh -opengl -fps=10 -benchmark  -carla-world-port=2000
#DISPLAY= simulator/CarlaUE4.sh -opengl -fps=10 -benchmark  -carla-world-port=3000
#DISPLAY= simulator/CarlaUE4.sh -opengl -fps=10 -benchmark  -carla-world-port=4000
#CUDA_VISIBLE_DEVICES=1 ./CarlaUE4.sh -fps=10 -benchmark  # this is untested

NAME=1280res_150ksamples
NAME=debug
NAME=1280res_150ksamples_noise
mkdir -p ./data/$NAME/
python data_collector.py --dataset_path=data/$NAME \
--frames_per_episode 2000 \
--n_episodes 5000 \
--nodisplay \

<< sample_cmd
bash ~/BEVSEG/lbc/scripts/collect_data.sh
cksbatch --nodelist=freddie ~/BEVSEG/lbc/scripts/collect_data.sh
sample_cmd
