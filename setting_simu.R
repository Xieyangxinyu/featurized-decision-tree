# Python
# n = c(50, 100, 150, 257)
# n = c(100, 200, 500, 1000)
setting_total <-
  expand.grid(name = c("linear", "rbf", "matern32", "complex"),
              n = c(100, 200, 500, 1000), 
              dim = c(25, 50, 100),
              i = 0:19, path = c("cat", "cont", "adult", "mi"))

names(setting_total) <- c("--dataset", "--n_obs", "--dim_in", "--rep", "--path")
write.csv(setting_total, "/Users/wdeng/Desktop/FDT/FDT/python/experiments/expr/args/settings.csv")


# remove previous bash script
file_handle <- "command_exec"
file_names <- 
  grep(file_handle, list.files(), value = TRUE)
file_names_bkp <- 
  paste0("./", file_names)

for (name in c(file_names, file_names_bkp)){
  file.remove(name)
}


# generate cluster bash script
n_run_per_worker <- 1
n_exec <- ceiling(nrow(setting_total) / (n_run_per_worker * 1e3))
setwd("/Users/wdeng/Desktop/FDT/FDT/python/experiments/expr/work")
for (exec_id in 1:n_exec) {
  n_rask_rest <- 
    min(nrow(setting_total)/n_run_per_worker - 
          (exec_id-1)*1e3, 1e3)
    string <- 
      paste0(
        "#!/bin/bash
#SBATCH --array=", (exec_id-1) * 1e3 + 1, 
        "-", (exec_id-1) * 1e3 + n_rask_rest, 
        "	    		#job array list goes 1,2,3...n
#SBATCH -c 1
#SBATCH -N 1
#SBATCH -t 0-04:00			#job run 12 hour
#SBATCH -p shared
#SBATCH --mem=6G
#SBATCH -o %A_%a.out
#SBATCH -e %A_%a.err
#SBATCH --mail-type=END      #Type of email notification
#SBATCH --mail-user=wdeng@g.harvard.edu
module load Anaconda3/2020.11

source activate RFDT

python ../../../src/experimenter.py --rowid ${SLURM_ARRAY_TASK_ID} --subproc ../../../src/var_importance/fdt_simu.py --args ../args/settings.csv --dir_out ../results

conda deactivate"
      )
  write(string, file = paste0("rff_exec", exec_id, ".sh"))
}







# R methods
setting_total4 <-
  expand.grid(name = c("linear", "rbf", "matern32", "complex"),
              n = c(100, 200, 500, 1000),
              dim = c(25, 50, 100),
              i = 0:19, data = "cat")

setting_total5 <-
  expand.grid(name = c("linear", "rbf", "matern32", "complex"),
              n = c(100, 200, 500, 1000),
              dim = c(25, 50, 100),
              i = 0:19, data = "cont")

setting_total1 <-
  expand.grid(name = c("linear", "rbf", "matern32", "complex"),
              n = c(100, 200, 500, 1000),
              dim = c(25, 50, 100),
              i = 0:19, data = "adult")

setting_total2 <-
  expand.grid(name = c("linear", "rbf", "matern32", "complex"),
              n = c(50, 100, 150, 257),
              dim = c(25, 50, 100),
              i = 0:19, data = "heart")

setting_total3 <-
  expand.grid(name = c("linear", "rbf", "matern32", "complex"),
              n = c(100, 200, 500, 1000),
              dim = c(25, 50, 100),
              i = 0:19, data = "mi")

setting_total <- rbind(setting_total4, setting_total5, setting_total1, setting_total2, setting_total3)
write.table(setting_total, "/Users/wdeng/Desktop/FDT/FDT/R/settings.txt", sep = ",", row.names = FALSE)


# remove previous bash script
file_handle <- "command_exec"
file_names <- 
  grep(file_handle, list.files(), value = TRUE)
file_names_bkp <- 
  paste0("./", file_names)

for (name in c(file_names, file_names_bkp)){
  file.remove(name)
}


# generate cluster bash script
n_run_per_worker <- 1
n_exec <- ceiling(nrow(setting_total) / (n_run_per_worker * 1e3))
setwd("/Users/wdeng/Desktop/FDT/FDT/R")
for (exec_id in 1:n_exec) {
  n_rask_rest <- 
    min(nrow(setting_total)/n_run_per_worker - 
          (exec_id-1)*1e3, 1e3)
  string <- 
    paste0(
      "#!/bin/bash
#SBATCH --array=", (exec_id-1) * 1e3 + 1, 
      "-", (exec_id-1) * 1e3 + n_rask_rest, 
      "	    		#job array list goes 1,2,3...n
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

Rscript './sim_comp.R' ${SLURM_ARRAY_TASK_ID}"
    )
  write(string, file = paste0("command_exec", exec_id, ".sh"))
}
