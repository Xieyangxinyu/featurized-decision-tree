setwd('featurized-decision-tree/R')
### Clear Environment ### 
rm(list = ls(all = TRUE))

### Load in the R libraries ###
library(coda)
library(MASS)
library(doParallel)
library(Rcpp)
library(RcppArmadillo)
library(BGLR)
library(monomvn)
library(kernlab)
library(adegenet)
library(magrittr)
library(ranger)
library(randomForest)
library(knockoff)
library(BNN)
library(bkmr)
library(pROC)
library(mgcv)

options(java.parameters = "-Xmx5000m")
library(bartMachine) # BART

# library(spikeSlabGAM) # BSTARSS 
library(earth) # MARS
# library(dplyr)
source('model_comparison.R')
sourceCpp("BAKRGibbs.cpp")
config_all <- read.csv("settings.txt")


# read config
print("Mr Handy: How may be of service, master?")
print("Mr Handy: Oh a 'argument', how wonderful..")
args <- commandArgs(trailingOnly = TRUE)
config_idx <- as.numeric(args)
# eval(parse(text = args))
print(sprintf("Mr Handy: You see the config index '%d'..", config_idx))
print("Mr Handy: and who gets to read all this mumble jumble? Me, that's who...")

# extract config and execute
setTable <- config_all[config_idx, ]

ord_varimp <- function(fit_res) {
  name_lst <- strsplit(names(fit_res), "x")
  ord <- NULL
  for (l in 1:length(name_lst)) {
    ord <- c(ord, as.numeric(name_lst[[l]][2]))
  }
  order(ord)
}

for (j in 1:nrow(setTable)){
  set.seed(0921)
  # extract command
  name <- setTable$name[j]
  n <- setTable$n[j]
  dim <- setTable$dim[j]
  i <- setTable$i[j]
  data_name <- setTable$data[j]
  
  # execute command
  taskname <- md_comp(name = name, n = n, dim = dim, i = i, data_name = data_name)
  
  # sign out sheet
  write(
    paste(config_idx, taskname, Sys.time(),
          collapse = "\t\t"),
    file = "sign_out.txt", append = TRUE)
  print("Mr Handy: there, 'sign_out.txt' signed. You hava a --")
}

print("Mr Handy: Here kitty-kitty-kitty....")
