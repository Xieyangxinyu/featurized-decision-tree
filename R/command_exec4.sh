#!/bin/bash
#SBATCH --array=3001-4000	    		#job array list goes 1,2,3...n
#SBATCH -c 1
#SBATCH -N 1
#SBATCH -t 0-12:00			#job run 12 hour
#SBATCH -p shared			#submit to 'short' queue
#SBATCH --mem=19370  		# use 4 GB memory
#SBATCH -o %A_%a.out
#SBATCH -e %A_%a.err
#SBATCH --mail-type=END      #Type of email notification
#SBATCH --mail-user=wdeng@g.harvard.edu
module load R/4.1.0-fasrc01
export R_LIBS_USER=$HOME/apps/R_4.1.0:$R_LIBS_USER

Rscript './sim_comp.R' ${SLURM_ARRAY_TASK_ID}
