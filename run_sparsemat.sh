#!/bin/bash
#SBATCH --time=36:00:00
#SBATCH --nodes=1     ### num nodes
#SBATCH --mem=24G
#SBATCH --ntasks-per-node=12
#SBATCH --account=pi-melamed

export OPENBLAS_MAIN_FREE=1
module load Anaconda3/5.3.0
#ipython skips.py 
python /project2/melamed/wrk/iptw/code/matchweight/his2ft.py $@
