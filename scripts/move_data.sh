#!/usr/bin/env bash

##################################################
##### Additional Instructions for Riselab Researchers
##################################################

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 2-00:00
#SBATCH --cpus-per-task 10
cd $CODE/BEVSEG/lbc

rsync_local_data_to_remote_data /data/ck/data/lbc/1280res_150ksamples/ freddie flaminio


##################################################
##### Instructions for General Researchers
##################################################




<< sample_cmd
bash ~/BEVSEG/lbc/scripts/collect_data.sh
cksbatch --nodelist=freddie ~/BEVSEG/lbc/scripts/collect_data.sh
sample_cmd
