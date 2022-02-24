#!/bin/bash
#SBATCH -p thin
#SBATCH -N 1 -c 128 --exclusive
#SBATCH -t 2-00:00:00

. ~/conda/etc/profile.d/conda.sh
conda activate correlated_hoppings
export OMP_NUM_THREADS=64

python3 correlated_hoppings/system.py "$@"
