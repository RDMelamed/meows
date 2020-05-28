#!/bin/bash
#SBATCH --time=36:00:00
#SBATCH --nodes=1     ### num nodes
#SBATCH --mem=30G
#SBATCH --ntasks-per-node=6
#SBATCH --account=pi-arzhetsky #
#SBATCH --partition=rzhetskyp100

## #SBATCH --account=pi-melamed #pi-arzhetsky #

module load Anaconda3/5.3.0
module load R
#ipython skips.py 
#/software/Anaconda3-5.0.0.1-el7-x86_64/bin/
python /project2/melamed/wrk/iptw/code/matchweight/run_steps.py $@
