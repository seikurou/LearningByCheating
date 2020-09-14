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
export PYTHONPATH="$CODE/BEVSEG/lbc/PythonAPI:$PYTHONPATH"

##################################################
##### Instructions for General Researchers
##################################################

NAME=original_teacher
#NAME=teacher_noshift_0

python benchmark_agent.py --suite=regular \
--model-path=./ckpts/$NAME/model-64.th \
--max-run 25 \
--port 3000 \

<< sample_cmd
#bash ~/BEVSEG/lbc/scripts/.sh
cksbatch --nodelist=freddie ~/BEVSEG/lbc/scripts/benchmark_teacher.sh
sample_cmd
