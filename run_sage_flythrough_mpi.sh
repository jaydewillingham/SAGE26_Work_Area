#!/bin/bash

#SBATCH --job-name=sage_flythrough_mpi
#SBATCH --output=slurm-flythrough-mpi-%A_%a.out
#SBATCH --array=0-3
#SBATCH --ntasks=6
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00
#SBATCH --mem-per-cpu=1GB

# MPI-parallelized flythrough rendering
# Each array task runs a different mode/color combination
# Each task uses 6 MPI ranks to parallelize frame rendering

# List of argument sets for each array job
ARGS_LIST=(
  "orbit density mov"
  "orbit type mov"
  "orbit mass mov"
  "orbit sfr mov"
)

ml purge
ml restore basic2

ARGS=${ARGS_LIST[$SLURM_ARRAY_TASK_ID]}
set -- $ARGS

MODE=$1
COLOR=$2
FORMAT=$3
START_SNAP=$4
END_SNAP=$5
NUM_ORBITS=10

echo "Running mode=$MODE color=$COLOR format=$FORMAT with $SLURM_NTASKS MPI ranks"

srun python plotting/flythrough.py --mode $MODE --color-by $COLOR --format $FORMAT --num-orbits $NUM_ORBITS