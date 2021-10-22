#!/bin/bash

#SBATCH --job-name capgmsfcn   ## name that will show up in the queue
#SBATCH --output ./slurm/slurm_out/%x-%j.out   ## filename of the output; the %j is equal to jobID; default is slurm-[jobID].out
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=10G
#SBATCH --partition=backfill  ## the partitions to run in (comma seperated)
##SBATCH --exclude=discovery-g[1,12,13]
##SBATCH --nodelist=discovery-g[12,13]
#SBATCH --time=0-24:00:00  ## time for analysis (day-hour:min:sec)

## Load modules
#module load anaconda3
#conda activate py37c111pip

pythonFile=run/run_maintrainer.py

time {
exp=5
model=MSFCN_SEVT_P2
batch_size=64
epoch=50
patience=10
lr=2e-3
nfold=5

for fold in 0; do
    for winsize in 1000; do
        for stride in 1; do
            set -x #echo on
            srun -N 1 -n 1 --exclusive -u python -u $pythonFile --model=$model --batch-size=32 --num-layer=3 --exp=$exp --fold=$fold \
                            --epochs=$epoch --patience=$patience --lr=$lr --nfold=$nfold --winsize=$winsize --stride=$stride &
            set +x
        done
    done
done


wait
}