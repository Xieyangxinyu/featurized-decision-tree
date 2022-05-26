#!/bin/bash
#SBATCH --array=1-960	    		#job array list goes 1,2,3...n
#SBATCH -c 1
#SBATCH -N 1
#SBATCH -t 0-02:00			#job run 12 hour
#SBATCH -p shared
#SBATCH --mem=80G
#SBATCH -o %A_%a.out
#SBATCH -e %A_%a.err
#SBATCH --mail-type=END      #Type of email notification
#SBATCH --mail-user=wdeng@g.harvard.edu
module load Anaconda3/2020.11

source activate RFDT

python ../../../src/experimenter.py --rowid ${SLURM_ARRAY_TASK_ID} --subproc ../../../src/var_importance/fdt_simu.py --args ../args/settings_cont.csv --dir_out ../results/cont

conda deactivate
