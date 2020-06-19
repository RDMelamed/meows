#!/bin/bash
#SBATCH --time=50:00:00
#SBATCH --partition=rzhetskyp100 #broadwl
#SBATCH --nodes=1     ### num nodes
#SBATCH --mem=50G
#SBATCH --ntasks-per-node=16
#SBATCH --account=pi-arzhetsky
#  no #SBATCH --account=pi-melamed

# export PATH=~/bin/anaconda/bin/:$PATH

export SQLITE_TMPDIR=/project2/arzhetsky/MSdb/tmp
export TMPDIR=/project2/arzhetsky/MSdb/tmp
export PYTHONPATH=/project2/melamed/wrk/iptw/code:$PYTHONPATH
module load Anaconda3/5.3.0
#ipython /project2/melamed/wrk/iptw/code/history_sparse_dense.py $@
#/software/Anaconda3-5.0.0.1-el7-x86_64/bin/ipython /project2/melamed/wrk/iptw/code/matchweight/history_sparse_dense.py $@

python /project2/melamed/wrk/iptw/code/matchweight/history_sparse_dense.py $@
