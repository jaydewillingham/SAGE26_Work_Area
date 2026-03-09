#!/bin/bash

#SBATCH --job-name=sage_flythrough
#SBATCH --output=slurm-flythrough-%A_%a.out
#SBATCH --array=0-1
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00
#SBATCH --mem=2G

# List of argument sets for each array job
ARGS_LIST=(
  "orbit density mov"
  "orbit type mov"
)

ml gcc/12.2.0
ml openmpi/4.1.4
ml gsl/2.7
ml fftw/3.3.10
ml hdf5/1.14.0
ml git/2.38.1

ml purge
ml restore basic

ARGS=${ARGS_LIST[$SLURM_ARRAY_TASK_ID]}
set -- $ARGS

MODE=$1

if [[ $MODE == "evolution" || $MODE == "combined" ]]; then
    python plotting/flythrough.py --mode $MODE  --color-by $2 --format $3
else
    python plotting/flythrough.py --mode $MODE --color-by $2 --format $3
fi