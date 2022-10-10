library(ggplot2)
library(dplyr)
library(reshape2)
library(MASS)
library(naniar)
library(gridExtra)

################################################ AUC: python ################################################ 
col_names <- c("data", "n", "dim", "rep", "path", "f1_fdt_c01", "f1_fdt_cat_c01", "roc_fdt_c01", 
               "roc_fdt_cat_c01", "f1_fdt_c1", "f1_fdt_cat_c1", "roc_fdt_c1", "roc_fdt_cat_c1", 
               "f1_extra", "f1_rf_raw", "roc_extra", "roc_rf_raw", "f1_rfnn", "roc_rfnn", "f1_nn",
               "roc_nn", "f1_nn_best", "roc_nn_best", "lengthscale", "tst_mse_fdt", "tst_mse_extra",
               "tst_mse_raw", "tst_mse_rfnn", "tst_mse_nn", "tst_mse_nn_best")
path_name <- paste0("featurized-decision-tree/python/experiments/expr/results/results.csv")
file <- read.csv(path_name, header = FALSE)
names(file) <- col_names 
file <- na.omit(file)
file <- filter(file, data %in% c("linear", "rbf", "matern32", "complex"))


file_auc <- file[, c(1:5, 8, 9, 16, 17, 19, 23)]
file_mse <- file[, c(1:5, 25:29)]


file_auc[, c(2:4, 6:11)] <- sapply(file_auc[, c(2:4, 6:11)], as.numeric)
file_mse[, c(2:4, 6:10)] <- sapply(file_mse[, c(2:4, 6:10)], as.numeric)

file_all_python <- file 

auc_grad <- NULL
auc_cst <- NULL
main_mse <- NULL

path <- "mi" # need to repeat for ("cat", "cont", "adult", "heart", "mi") from line 35 to line 237

cur_auc <- file_auc[which(file_auc$path == path), ]
names(cur_auc)[6:11] <- c("FDT", "FDT_cat", "Extra", "RF", "RFF", "NN")
tmp <- summarise_at(group_by(cur_auc, data, n, dim), vars(FDT, FDT_cat, Extra, RF, RFF, NN), 
                    list(~mean(., na.rm = TRUE)))
tmp1 <- melt(tmp, id.vars = c("data", "n", "dim"), variable.name = "method", value.name = "auc") 
tmp <- summarise_at(group_by(cur_auc, data, n, dim), vars(FDT, FDT_cat, Extra, RF, RFF, NN), 
                    list(~sd(., na.rm = TRUE)))
tmp2 <- melt(tmp, id.vars = c("data", "n", "dim"), variable.name = "method", value.name = "sd")
file0 <- merge(tmp1, tmp2)
python_auc <- file0 
python_auc <- filter(python_auc, dim %in% c(25, 50, 100, 200))
python_auc <- filter(python_auc, data %in% c("linear", "rbf", "matern32", "complex"))


cur_mse <- file_mse[which(file_mse$path == path), ]
names(cur_mse)[6:10] <- c("FDT", "Extra", "RF", "RFF", "NN")
tmp <- summarise_at(group_by(cur_mse, data, n, dim), vars(FDT, Extra, RF, RFF, NN), 
                    list(~mean(., na.rm = TRUE)))
tmp1 <- melt(tmp, id.vars = c("data", "n", "dim"), variable.name = "method", value.name = "mse") 
tmp <- summarise_at(group_by(cur_mse, data, n, dim), vars(FDT, Extra, RF, RFF, NN), 
                    list(~sd(., na.rm = TRUE)))
tmp2 <- melt(tmp, id.vars = c("data", "n", "dim"), variable.name = "method", value.name = "sd")
file0 <- merge(tmp1, tmp2)
python_mse <- file0 
python_mse <- filter(python_mse, dim %in% c(25, 50, 100, 200))
python_mse <- filter(python_mse, data %in% c("linear", "rbf", "matern32", "complex"))
# 7 x 9 


################################################ R_methods ################################################ 
path_name <- paste0("featurized-decision-tree/R/sim_auc_", path, ".txt")
file <- read.table(path_name, fill = TRUE)
names(file) <- c("data", "n", "dim", "rep", "Knockoff", "BNN", "BKMR", 
                 "BART", "GAM", "MARS", "BAKR", "BRR", "BL")
file <- filter(file, dim %in% c(25, 50, 100, 200))
file <- na.omit(file)
file[, 2:13] <- sapply(file[, 2:13], as.numeric)
file <- na.omit(file)

file0 <- replace_with_na(file, replace = list(BNN = -1))
file0 <- replace_with_na(file0, replace = list(GAM = -1))

tmp <- summarise_at(group_by(file0, data, n, dim), vars(Knockoff, BNN, BKMR, BART, GAM, BAKR, BRR, BL), 
                    list(~mean(., na.rm = TRUE)))
tmp1 <- melt(tmp, id.vars = c("data", "n", "dim"), variable.name = "method", value.name = "auc") 
tmp <- summarise_at(group_by(file0, data, n, dim), vars(Knockoff, BNN, BKMR, BART, GAM, BAKR, BRR, BL), 
                    list(~sd(., na.rm = TRUE)))
tmp2 <- melt(tmp, id.vars = c("data", "n", "dim"), variable.name = "method", value.name = "sd")
file0 <- merge(tmp1, tmp2)
R_auc <- file0 
R_auc <- filter(R_auc, data %in% c("linear", "rbf", "matern32", "complex"))


path_name <- paste0("featurized-decision-tree/R/sim_mse_", path, ".txt")
file <- read.table(path_name, fill = TRUE)
names(file) <- c("data", "n", "dim", "rep", "BKMR", "MARS", "BART", "GAM", "BAKR", "BRR", "BL")
file <- filter(file, dim %in% c(25, 50, 100, 200))
file <- na.omit(file)

file[, 2:11] <- sapply(file[, 2:11], as.numeric)
file <- na.omit(file)

file0 <- replace_with_na(file, replace = list(BKMR = -1))
file0 <- replace_with_na(file0, replace = list(GAM = -1))

tmp <- summarise_at(group_by(file0, data, n, dim), vars(BKMR, BART, GAM, MARS, BAKR, BRR, BL), 
                    list(~mean(., na.rm = TRUE)))
tmp1 <- melt(tmp, id.vars = c("data", "n", "dim"), variable.name = "method", value.name = "mse")  
tmp <- summarise_at(group_by(file0, data, n, dim), vars(BKMR, BART, GAM, MARS, BAKR, BRR, BL), 
                    list(~sd(., na.rm = TRUE)))
tmp2 <- melt(tmp, id.vars = c("data", "n", "dim"), variable.name = "method", value.name = "sd")
file0 <- merge(tmp1, tmp2)
R_mse <- file0
R_mse <- filter(R_mse, data %in% c("linear", "rbf", "matern32", "complex"))

################################################ AUC: all ################################################ 
## gradient estimates
python_method <- filter(python_auc, method %in% c("FDT", "Extra", "RFF", "NN"))
python_method[python_method$method == "Extra", "method"] <- "RF"
file_all <- rbind(python_method, R_auc)
file_all$method <- as.character(file_all$method)
tmp <- filter(file_all, (data == "matern32") & (dim == 100))
tmp$data <- ifelse(path == "cat", "mixture", ifelse(path == "cont", "continuous", path))
auc_grad <- rbind(auc_grad, tmp)

file <- file_all
file <- filter(file, method %in% c("FDT", "RFF", "RF", "Knockoff", "BKMR", "BART", 
                                   "GAM", "BAKR", "BRR", "BL", "NN"))
file$data <- factor(file$data, order = T, levels = c("linear", "rbf", "matern32", "complex"))

file[file$method == "FDT", "method"] <- "RF-FDT (Ours)"
file[file$method == "RF", "method"] <- "RF-Impurity"
file[file$method == "Knockoff", "method"] <- "RF-KnockOff"
file[file$method == "RFF", "method"] <- "RFF (Ours)"
file[file$method == "NN", "method"] <- "NN (Ours)" 
file[file$method == "GAM", "method"] <- "GAM (Ours)"
file$method <- factor(file$method, order = T, levels = c("RF-FDT (Ours)", "NN (Ours)", "GAM (Ours)", 
                                                         "RF-Impurity", "RFF (Ours)", "BRR", "RF-KnockOff", 
                                                         "BKMR", "BL", "BART", "BAKR"))
p1 <- ggplot(data = file, aes(x = n, y = auc, color = method, linetype = method))+ 
  scale_colour_manual(values = c("purple", "orange", "brown", "blue", "darkgreen", "deeppink2",        
                                 "skyblue", "darkolivegreen3", "brown1", "darkslateblue", "chartreuse4")) + 
  scale_linetype_manual(values = c("solid", "solid", "solid", "dashed", "solid", "dashed", 
                                   "dotdash", "dashed", "dotted", "dotted", "dotted"))+
  geom_line() + facet_grid(dim ~ data) + labs(x = "n", y = 'AUROC') + 
  geom_point(size = .5) + theme_set(theme_bw()) + ylim(0, 1) + 
  theme(axis.title.x = element_text(size = 15), axis.title.y = element_text(size = 15),
        plot.title = element_text(size = 16), axis.text.x=element_text(size = 12),
        axis.text.y=element_text(size = 12)) + theme(legend.position = "top") + 
  theme(legend.title = element_text(size = 12, face = "bold")) +
  theme(legend.text = element_text(size = 12))
ggsave(paste0("featurized-decision-tree/plots/", path, "_auc_grad.pdf"), width = 9, height = 7)


## contrast for categorical covariates
python_method <- filter(python_auc, method %in% c("FDT_cat", "Extra", "RFF", "NN"))
python_method[python_method$method == "Extra", "method"] <- "RF"
file_all <- rbind(python_method, R_auc)
file_all$method <- as.character(file_all$method)
tmp <- filter(file_all, (data == "matern32") & (dim == 100))
tmp$data <- ifelse(path == "cat", "mixture", ifelse(path == "cont", "continuous", path))
auc_cst <- rbind(auc_cst, tmp)

file <- file_all
file[file$method == "FDT_cat", "method"] <- "FDT"
file <- filter(file, method %in% c("FDT", "RFF", "RF", "Knockoff", "BKMR", "BART", 
                                   "GAM", "BAKR", "BRR", "BL", "NN"))
file$data <- factor(file$data, order = T, levels = c("linear", "rbf", "matern32", "complex"))
file[file$method == "FDT", "method"] <- "RF-FDT (Ours)"
file[file$method == "RF", "method"] <- "RF-Impurity"
file[file$method == "Knockoff", "method"] <- "RF-KnockOff"
file[file$method == "RFF", "method"] <- "RFF (Ours)"
file[file$method == "NN", "method"] <- "NN (Ours)"
file[file$method == "GAM", "method"] <- "GAM (Ours)"
file$method <- factor(file$method, order = T, levels = c("RF-FDT (Ours)", "NN (Ours)", "GAM (Ours)", 
                                                         "RF-Impurity", "RFF (Ours)", "BRR", "RF-KnockOff", 
                                                         "BKMR", "BL", "BART", "BAKR"))
p2 <- ggplot(data = file, aes(x = n, y = auc, color = method, linetype = method))+ 
  scale_colour_manual(values = c("purple", "orange", "brown", "blue", "darkgreen", "deeppink2",        
                                 "skyblue", "darkolivegreen3", "brown1", "darkslateblue", "chartreuse4")) + 
  scale_linetype_manual(values = c("solid", "solid", "solid", "dashed", "solid", "dashed", 
                                   "dotdash", "dashed", "dotted", "dotted", "dotted"))+
  geom_line() + facet_grid(dim ~ data) + labs(x = "n", y = 'AUROC') + 
  geom_point(size = .5) + theme_set(theme_bw()) + ylim(0, 1) + 
  theme(axis.title.x = element_text(size = 15), axis.title.y = element_text(size = 15),
        plot.title = element_text(size = 16), axis.text.x=element_text(size = 12),
        axis.text.y=element_text(size = 12)) + theme(legend.position = "top") + 
  theme(legend.title = element_text(size = 12, face = "bold")) +
  theme(legend.text = element_text(size = 12))
ggsave(paste0("featurized-decision-tree/plots/", path, "_auc_cst.pdf"), width = 9, height = 7)
# 7 x 9



################################################ TST_MSE: all ################################################ 
file_all <- rbind(python_mse, R_mse)
file_all$method <- as.character(file_all$method)

file <- file_all
file$data <- factor(file$data, order = T, levels = c("linear", "rbf", "matern32", "complex"))

file <- filter(file, method %in% c("Extra", "RFF", "BKMR", "BART",
                                   "GAM", "BAKR", "BRR", "BL", "NN"))
file[file$method == "Extra", "method"] <- "FDT"

tmp <- filter(file, (data == "matern32") & (dim == 100))
tmp$data <- ifelse(path == "cat", "mixture", ifelse(path == "cont", "continuous", path))
main_mse <- rbind(main_mse, tmp)

nan_n <- if (path == "heart") c(50, 100, 150, 257) else c(100, 200, 500, 1000)
nan_data <- ifelse(path == "cat", "mixture", ifelse(path == "cont", "continuous", path))
mse_nan <- data.frame(data = nan_data, n = rep(nan_n, 2), dim = 100, 
                      method = rep(c("RF-Impurity", "RF-KnockOff"), each = 4), 
                      mse = NA, sd = NA)
main_mse <- rbind(main_mse, mse_nan)

file$dim <- factor(file$dim, order = T, levels = c(25, 50, 100, 200))
file$n <- as.numeric(file$n)

file[file$method == "FDT", "method"] <- "RF-FDT (Ours)"
file[file$method == "RFF", "method"] <- "RFF (Ours)"
file[file$method == "GAM", "method"] <- "GAM (Ours)"
file[file$method == "NN", "method"] <- "NN (Ours)"

file$method <- factor(file$method, order = T, levels = c("RF-FDT (Ours)", "NN (Ours)", "GAM (Ours)", 
                                                         "RFF (Ours)", "BRR", "BKMR", "BL", "BART", "BAKR"))

p3 <- ggplot(data = file, aes(x = n, y = mse, color = method, linetype = method))+ 
  scale_colour_manual(values = c("purple", "orange", "brown", "darkgreen", "deeppink2",        
                                 "darkolivegreen3", "brown1", "darkslateblue", "chartreuse4")) + 
  scale_linetype_manual(values = c("solid", "solid", "solid", "solid", "dashed", 
                                   "dashed", "dotted", "dotted", "dotted"))+
  scale_y_continuous(trans='log10') + 
  geom_line() + facet_grid(dim ~ data) + labs(x = "n", y = 'Testing MSE') + 
  geom_point(size = .5) + theme_set(theme_bw()) + 
  theme(axis.title.x = element_text(size = 15), axis.title.y = element_text(size = 15),
        plot.title = element_text(size = 16), axis.text.x=element_text(size = 12),
        axis.text.y=element_text(size = 12)) + theme(legend.position = "top") + 
  theme(legend.title = element_text(size = 12, face = "bold")) +
  theme(legend.text = element_text(size = 12))
ggsave(paste0("featurized-decision-tree/plots/", path, "_mse.pdf"), width = 9, height = 7)




################################################ main ################################################ 
dd <- filter(auc_grad, data == "continuous")
auc_cst <- filter(auc_cst, data != "continuous")
auc_cst <- rbind(auc_cst, dd)
auc_grad[auc_grad$data == "mixture", "data"] <- "synthetic-mixture"
auc_grad[auc_grad$data == "continuous", "data"] <- "synthetic-continuous"
auc_cst[auc_cst$data == "mixture", "data"] <- "synthetic-mixture"
auc_cst[auc_cst$data == "continuous", "data"] <- "synthetic-continuous"
auc_cst[auc_cst$method == "FDT_cat", "method"] <- "FDT"
main_mse[main_mse$data == "mixture", "data"] <- "synthetic-mixture"
main_mse[main_mse$data == "continuous", "data"] <- "synthetic-continuous"


get_legend<-function(myggplot){
  tmp <- ggplot_gtable(ggplot_build(myggplot))
  leg <- which(sapply(tmp$grobs, function(x) x$name) == "guide-box")
  legend <- tmp$grobs[[leg]]
  return(legend)
}

auc_cst$type <- "AUC"
main_mse$type <- "Testing MSE"
names(auc_cst)[5] <- "value"
names(main_mse)[5] <- "value"

auc_grad$type <- "AUC"
names(auc_grad)[5] <- "value"


file <- rbind(auc_cst, main_mse)
file$n <- as.integer(file$n)
file <- filter(file, method %in% c("FDT", "RFF", "RF", "Knockoff", "BKMR", "BART", 
                                   "GAM", "BAKR", "BRR", "BL", "NN"))
file[file$method == "FDT", "method"] <- "RF-FDT (Ours)"
file[file$method == "RF", "method"] <- "RF-Impurity"
file[file$method == "Knockoff", "method"] <- "RF-KnockOff"
file[file$method == "RFF", "method"] <- "RFF (Ours)"
file[file$method == "GAM", "method"] <- "GAM (Ours)"
file[file$method == "NN", "method"] <- "NN (Ours)"


file$method <- factor(file$method, order = T, levels = c("RF-FDT (Ours)", "NN (Ours)", "GAM (Ours)", 
                                                         "RF-Impurity", "RFF (Ours)", "BRR", "RF-KnockOff", 
                                                         "BKMR", "BL", "BART", "BAKR"))

file$data <- factor(file$data, order = T, levels = c("synthetic-mixture", "synthetic-continuous", 
                                                     "adult", "heart", "mi"))

  
fdt0 <- ggplot(filter(file, (data == "synthetic-mixture") & (type == "AUC")), 
               aes(x = n, y = value, color = method, linetype = method))+ 
  scale_colour_manual(values = c("purple", "orange", "brown", "blue", "darkgreen", "deeppink2",        
                                 "skyblue", "darkolivegreen3", "brown1", "darkslateblue", "chartreuse4")) + 
  scale_linetype_manual(values = c("solid", "solid", "solid", "dashed", "solid", "dashed", 
                                   "dotdash", "dashed", "dotted", "dotted", "dotted"))+
  scale_y_continuous(trans='log10') + 
  geom_line() + labs(x = "n", y = "Testing MSE") + 
  geom_point(size = .2) + theme_set(theme_bw()) + ggtitle("FDT") + 
  theme(plot.title = element_text(hjust = 0.5), 
        axis.title.x=element_blank(), axis.text.x=element_blank(), axis.ticks.x=element_blank(), 
        axis.title.y = element_blank(), legend.position = "top", legend.title = element_text(size=14), 
        legend.text = element_text(size=14))
legend <- get_legend(fdt0)


q1 <- ggplot(filter(file, (data == "synthetic-mixture") & (type == "Testing MSE")), 
               aes(x = n, y = value, color = method, linetype = method))+ 
  scale_colour_manual(values = c("purple", "orange", "brown", "darkgreen", "deeppink2",
                                 "darkolivegreen3", "brown1", "darkslateblue", "chartreuse4")) +
    scale_linetype_manual(values = c("solid", "solid", "solid", "solid", "dashed",
                                     "dashed", "dotted", "dotted", "dotted"))+
  ylim(0, 3) + 
  geom_line() + labs(x = "n", y = "Testing MSE") + 
  geom_point(size = .2) + theme_set(theme_bw()) + 
  theme(axis.title.x=element_blank()) + theme(legend.position = "none")

q2 <- ggplot(filter(file, (data == "synthetic-continuous") & (type == "Testing MSE")), 
             aes(x = n, y = value, color = method, linetype = method))+ 
  scale_colour_manual(values = c("purple", "orange", "brown", "darkgreen", "deeppink2",
                                 "darkolivegreen3", "brown1", "darkslateblue", "chartreuse4")) +
  scale_linetype_manual(values = c("solid", "solid", "solid", "solid", "dashed",
                                   "dashed", "dotted", "dotted", "dotted"))+
  ylim(0, 3) + 
  geom_line() + labs(x = "n", y = "Testing MSE") + 
  geom_point(size = .2) + theme_set(theme_bw()) + 
  theme(axis.title.x=element_blank(), axis.title.y=element_blank()) + theme(legend.position = "none")

q3 <- ggplot(filter(file, (data == "adult") & (type == "Testing MSE")), 
             aes(x = n, y = value, color = method, linetype = method))+ 
  scale_colour_manual(values = c("purple", "orange", "brown", "darkgreen", "deeppink2",
                                 "darkolivegreen3", "brown1", "darkslateblue", "chartreuse4")) +
  scale_linetype_manual(values = c("solid", "solid", "solid", "solid", "dashed",
                                   "dashed", "dotted", "dotted", "dotted"))+
  ylim(0, 3) + 
  geom_line() + labs(x = "n", y = "Testing MSE") + 
  geom_point(size = .2) + theme_set(theme_bw()) + 
  theme(axis.title.x=element_blank(), axis.title.y=element_blank()) + theme(legend.position = "none")


q4 <- ggplot(filter(file, (data == "heart") & (type == "Testing MSE")), 
             aes(x = n, y = value, color = method, linetype = method))+ 
  scale_colour_manual(values = c("purple", "orange", "brown", "darkgreen", "deeppink2",
                                 "darkolivegreen3", "brown1", "darkslateblue", "chartreuse4")) +
  scale_linetype_manual(values = c("solid", "solid", "solid", "solid", "dashed",
                                   "dashed", "dotted", "dotted", "dotted"))+
  ylim(0, 3) + 
  geom_line() + labs(x = "n", y = "Testing MSE") + 
  geom_point(size = .2) + theme_set(theme_bw()) + 
  theme(axis.title.x=element_blank(), axis.title.y=element_blank()) + theme(legend.position = "none")


q5 <- ggplot(filter(file, (data == "mi") & (type == "Testing MSE")), 
             aes(x = n, y = value, color = method, linetype = method))+ 
  scale_colour_manual(values = c("purple", "orange", "brown", "darkgreen", "deeppink2",
                                 "darkolivegreen3", "brown1", "darkslateblue", "chartreuse4")) +
  scale_linetype_manual(values = c("solid", "solid", "solid", "solid", "dashed",
                                   "dashed", "dotted", "dotted", "dotted"))+
  ylim(0, 3) + 
  geom_line() + labs(x = "n", y = "Testing MSE") + 
  geom_point(size = .2) + theme_set(theme_bw()) + 
  theme(axis.title.x=element_blank(), axis.title.y=element_blank()) + theme(legend.position = "none")


q6 <- ggplot(filter(file, (data == "synthetic-mixture") & (type == "AUC")), 
             aes(x = n, y = value, color = method, linetype = method))+ 
  scale_colour_manual(values = c("purple", "orange", "brown", "blue", "darkgreen", "deeppink2",        
                                 "skyblue", "darkolivegreen3", "brown1", "darkslateblue", "chartreuse4")) + 
  scale_linetype_manual(values = c("solid", "solid", "solid", "dashed", "solid", "dashed", 
                                   "dotdash", "dashed", "dotted", "dotted", "dotted"))+
  geom_line() + labs(x = "n", y = "AUROC") + ggtitle("synthetic-mixture") + 
  geom_point(size = .2) + theme_set(theme_bw()) + 
  theme(plot.title = element_text(hjust = 0.5, size = 12, face = "bold")) +
  theme(axis.title.x=element_blank()) + theme(legend.position = "none")


q7 <- ggplot(filter(file, (data == "synthetic-continuous") & (type == "AUC")), 
             aes(x = n, y = value, color = method, linetype = method))+ 
  scale_colour_manual(values = c("purple", "orange", "brown", "blue", "darkgreen", "deeppink2",        
                                 "skyblue", "darkolivegreen3", "brown1", "darkslateblue", "chartreuse4")) + 
  scale_linetype_manual(values = c("solid", "solid", "solid", "dashed", "solid", "dashed", 
                                   "dotdash", "dashed", "dotted", "dotted", "dotted"))+
  geom_line() + labs(x = "n", y = "AUROC") + ggtitle("synthetic-continuous") + 
  geom_point(size = .2) + theme_set(theme_bw()) + 
  theme(plot.title = element_text(hjust = 0.5, size = 12, face = "bold")) +
  theme(axis.title.x=element_blank(), axis.title.y=element_blank()) + theme(legend.position = "none")


q8 <- ggplot(filter(file, (data == "adult") & (type == "AUC")), 
             aes(x = n, y = value, color = method, linetype = method))+ 
  scale_colour_manual(values = c("purple", "orange", "brown", "blue", "darkgreen", "deeppink2",        
                                 "skyblue", "darkolivegreen3", "brown1", "darkslateblue", "chartreuse4")) + 
  scale_linetype_manual(values = c("solid", "solid", "solid", "dashed", "solid", "dashed", 
                                   "dotdash", "dashed", "dotted", "dotted", "dotted"))+
  geom_line() + labs(x = "n", y = "AUROC") + ggtitle("adult") + 
  geom_point(size = .2) + theme_set(theme_bw()) + 
  theme(plot.title = element_text(hjust = 0.5, size = 12, face = "bold")) +
  theme(axis.title.x=element_blank(), axis.title.y=element_blank()) + theme(legend.position = "none")


q9 <- ggplot(filter(file, (data == "heart") & (type == "AUC")), 
             aes(x = n, y = value, color = method, linetype = method))+ 
  scale_colour_manual(values = c("purple", "orange", "brown", "blue", "darkgreen", "deeppink2",        
                                 "skyblue", "darkolivegreen3", "brown1", "darkslateblue", "chartreuse4")) + 
  scale_linetype_manual(values = c("solid", "solid", "solid", "dashed", "solid", "dashed", 
                                   "dotdash", "dashed", "dotted", "dotted", "dotted"))+
  geom_line() + labs(x = "n", y = "AUROC") + ggtitle("heart") + 
  geom_point(size = .2) + theme_set(theme_bw()) + 
  theme(plot.title = element_text(hjust = 0.5, size = 12, face = "bold")) +
  theme(axis.title.x=element_blank(), axis.title.y=element_blank()) + theme(legend.position = "none")

q10 <- ggplot(filter(file, (data == "mi") & (type == "AUC")), 
             aes(x = n, y = value, color = method, linetype = method))+ 
  scale_colour_manual(values = c("purple", "orange", "brown", "blue", "darkgreen", "deeppink2",        
                                 "skyblue", "darkolivegreen3", "brown1", "darkslateblue", "chartreuse4")) + 
  scale_linetype_manual(values = c("solid", "solid", "solid", "dashed", "solid", "dashed", 
                                   "dotdash", "dashed", "dotted", "dotted", "dotted"))+
  geom_line() + labs(x = "n", y = "AUROC") + ggtitle("mi") + 
  geom_point(size = .2) + theme_set(theme_bw()) + 
  theme(plot.title = element_text(hjust = 0.5, size = 12, face = "bold")) +
  theme(axis.title.x=element_blank(), axis.title.y=element_blank()) + theme(legend.position = "none")


grid.arrange(legend, q6, q7, q8, q9, q10, q1, q2, q3, q4, q5, ncol=5, nrow=3,
             layout_matrix = rbind(c(1,1,1,1,1), c(2,3,4,5,6), c(7,8,9,10,11)),
             widths = c(3.2, 3, 3, 3, 3), heights = c(1.0, 2.5, 2.3), bottom="n")

# 6 x 13



### tables
# library(reshape2)
# cur_file <- filter(file, data == "mi")
# dd <- cur_file[, c(2, 4, 5)]
# ssMean <- dcast(dd, n ~ method)
# 
# dd <- cur_file[, c(2, 4, 6)]
# ssSd <- dcast(dd, n ~ method)
# 
# dfN <- ssMean
# dfN[-1] <- paste0(round(as.matrix(ssMean[-1]), 2), " (", 
#                   round(as.matrix(ssSd[-1]), 2), ")")
# 
# write.csv(dfN, "/Users/wdeng/Desktop/dfN.csv", row.names = FALSE, quote = FALSE)


