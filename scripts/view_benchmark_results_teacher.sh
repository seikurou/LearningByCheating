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
NAME=teacher_3
#NAME=teacher_noshift_0

#python $CODE/BEVSEG/lbc/view_benchmark_results.py /data/ck/BEVSEG/lbc/ckpts/$NAME/benchmark/model-32
python $CODE/BEVSEG/lbc/view_benchmark_results.py /data/ck/BEVSEG/lbc/ckpts/$NAME/benchmark/model-64

<< sample_cmd
cksbatch --nodelist=freddie ~/BEVSEG/lbc/scripts/view_benchmark_results_teacher.sh
bash ~/BEVSEG/lbc/scripts/view_benchmark_results_teacher.sh
sample_cmd
