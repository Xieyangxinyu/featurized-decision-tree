setwd('/Users/wdeng/Desktop')
library(ggplot2)
library(dplyr)
library(reshape2)
library(MASS)
library(naniar)
library(gridExtra)

main_grad <- NULL
main_cst <- NULL
main_mse <- NULL

path <- "mi"
################################################ AUC: python ################################################ 
col_names <- c("data", "n", "dim", "rep", "path", "c", "FDT", "FDT_cat", "Extra", "RF", "RFNN",
               "lengthscale", "tst_mse_FDT", "tst_mse_Extra", "tst_mse_RF", "tst_mse_RFNN")
path_name <- paste0("/Users/wdeng/Desktop/FDT/results/20220518/wo_int/", path, "/results.csv")
file <- read.csv(path_name, header = TRUE)
# names(file) <- col_names 
file <- na.omit(file)
file <- filter(file, data %in% c("linear", "rbf", "matern32", "complex"))
file_auc <- file[, c(1:4, 7:11)]
file_auc$RFNN <- as.numeric(file_auc$RFNN)
tmp <- summarise_at(group_by(file_auc, data, n, dim), vars(FDT, FDT_cat, Extra, RF, RFNN), 
                    list(~mean(., na.rm = TRUE)))
tmp1 <- melt(tmp, id.vars = c("data", "n", "dim"), variable.name = "method", value.name = "auc") 
tmp <- summarise_at(group_by(file_auc, data, n, dim), vars(FDT, FDT_cat, Extra, RF, RFNN), 
                    list(~sd(., na.rm = TRUE)))
tmp2 <- melt(tmp, id.vars = c("data", "n", "dim"), variable.name = "method", value.name = "sd")
file0 <- merge(tmp1, tmp2)
file0$data <- factor(file0$data, order = T, levels = c("linear", "rbf", "matern32", "complex"))
fdt_file <- file0 
fdt_file <- filter(fdt_file, dim %in% c(25, 50, 100))


file_mse <- file[, c(1:4, 13:16)]
file_mse$tst_mse_RFNN <- as.numeric(file_mse$tst_mse_RFNN)
file_mse$tst_mse_RF <- as.numeric(file_mse$tst_mse_RF)
tmp <- summarise_at(group_by(file_mse, data, n, dim), 
                    vars(tst_mse_FDT, tst_mse_Extra, tst_mse_RF, tst_mse_RFNN), 
                    list(~mean(., na.rm = TRUE)))
tmp1 <- melt(tmp, id.vars = c("data", "n", "dim"), variable.name = "method", value.name = "tst_mse")
tmp <- summarise_at(group_by(file_mse, data, n, dim), 
                    vars(tst_mse_FDT, tst_mse_Extra, tst_mse_RF, tst_mse_RFNN), 
                    list(~sd(., na.rm = TRUE)))
tmp2 <- melt(tmp, id.vars = c("data", "n", "dim"), variable.name = "method", value.name = "sd")

file0_mse <- merge(tmp1, tmp2)
file0_mse$method <- as.character(file0_mse$method)
file0_mse[file0_mse$method == "tst_mse_FDT", "method"] <- "FDT"
file0_mse[file0_mse$method == "tst_mse_Extra", "method"] <- "Extra"
file0_mse[file0_mse$method == "tst_mse_RF", "method"] <- "RF"
file0_mse[file0_mse$method == "tst_mse_RFNN", "method"] <- "RFNN"
fdt_file_mse <- file0_mse
fdt_file_mse <- filter(fdt_file_mse, dim %in% c(25, 50, 100))

file0$dim <- factor(file0$dim, order = T, levels = c(25, 50, 100))
# ggplot(data = file0, aes(x = n, y = auc, color = method)) +
#   scale_colour_manual(values = c("purple", "darkgreen", "orange", "skyblue", "gray"))+
#   geom_line() + facet_grid(dim ~ data) + labs(x = "n", y = 'AUROC') +
#   geom_point(size = .5) + theme_set(theme_bw()) +
#   theme(axis.title.x = element_text(size = 15), axis.title.y = element_text(size = 15),
#         plot.title = element_text(size = 16), axis.text.x=element_text(size = 12),
#         axis.text.y=element_text(size = 12)) + theme(legend.position = "top") +
#   theme(legend.title = element_text(size = 12, face = "bold")) +
#   theme(legend.text = element_text(size = 12))

# 7 x 9 

# without MARS
################################################ AUC: R_methods ################################################ 
path_name <- paste0("/Users/wdeng/Desktop/FDT/results/20220518/wo_int/", path, "/sim_auc_", path, ".txt")
file <- read.table(path_name, fill = TRUE)
names(file) <- c("data", "n", "dim", "rep", "Knockoff", "BNN", "BKMR", 
                 "BART", "GAM", "MARS", "BAKR", "BRR", "BL")
file$n <- as.integer(file$n)
file$dim <- as.integer(file$dim)
file$rep <- as.integer(file$rep)
file <- na.omit(file)
file0 <- filter(file, data %in% c("linear", "rbf", "matern32", "complex"))
file0 <- filter(file0, dim %in% c(25, 50, 100))

col_convert <- c("Knockoff", "BNN", "BKMR", "BART", "GAM", "MARS", "BAKR", "BRR", "BL")
file0[col_convert] <- sapply(file0[col_convert], as.numeric) # a few warnings
file0 <- replace_with_na(file0, replace = list(BNN = -1))
file0 <- replace_with_na(file0, replace = list(GAM = -1))

tmp <- summarise_at(group_by(file0, data, n, dim), vars(Knockoff, BNN, BKMR, BART, GAM, BAKR, BRR, BL), 
                    list(~mean(., na.rm = TRUE)))
tmp1 <- melt(tmp, id.vars = c("data", "n", "dim"), variable.name = "method", value.name = "auc") 
tmp <- summarise_at(group_by(file0, data, n, dim), vars(Knockoff, BNN, BKMR, BART, GAM, BAKR, BRR, BL), 
                    list(~sd(., na.rm = TRUE)))
tmp2 <- melt(tmp, id.vars = c("data", "n", "dim"), variable.name = "method", value.name = "sd")
file0 <- merge(tmp1, tmp2)
file0 <- filter(file0, method %in% c("Knockoff","BNN", "BKMR", "BART", "GAM", "BAKR", "BRR", "BL"))
file0$data <- factor(file0$data, order = T, levels = c("linear", "rbf", "matern32", "complex"))
R_methods <- file0


################################################ AUC: all ################################################ 
## gradient estimates
fdt_method <- filter(fdt_file, method %in% c("FDT", "Extra", "RFNN"))
fdt_method[fdt_method$method == "Extra", "method"] <- "RF"
file_all <- rbind(fdt_method, R_methods)
file_all$method <- as.character(file_all$method)
tmp <- filter(file_all, (data == "matern32") & (dim == 100))
tmp$data <- ifelse(path == "cat", "mixture", ifelse(path == "cont", "continuous", path))
main_grad <- rbind(main_grad, tmp)

file <- file_all
file <- filter(file, method %in% c("FDT", "RFNN", "RF", "Knockoff", "BKMR", "BART", 
                                   "GAM", "BAKR", "BRR", "BL"))
file$data <- factor(file$data, order = T, levels = c("linear", "rbf", "matern32", "complex"))

file[file$method == "FDT", "method"] <- "RF-FDT (Ours)"
file[file$method == "RF", "method"] <- "RF-Impurity"
file[file$method == "BART", "method"] <- "BART"
file[file$method == "Knockoff", "method"] <- "RF-KnockOff"
file[file$method == "RFNN", "method"] <- "RFNN (Ours)"
file[file$method == "GAM", "method"] <- "GAM (Ours)"
file$method <- factor(file$method, order = T, levels = c("RF-FDT (Ours)", "RFNN (Ours)", 
                                                         "RF-Impurity", "BKMR", "BART", "BAKR", 
                                                         "RF-KnockOff",  "GAM (Ours)", "BRR", "BL"))
p1 <- ggplot(data = file, aes(x = n, y = auc, color = method, linetype = method))+ 
  scale_colour_manual(values = c("purple", "darkgreen", "blue", "darkolivegreen3", "darkslateblue", 
                                 "chartreuse4", "skyblue", "brown", "deeppink2", "brown1")) + 
  scale_linetype_manual(values = c("solid", "solid", "dashed", "dashed", "dotted", 
                                   "dotted", "dotdash", "solid", "dashed", "dotted"))+
  geom_line() + facet_grid(dim ~ data) + labs(x = "n", y = 'AUROC') + 
  geom_point(size = .5) + theme_set(theme_bw()) + ylim(0, 1) + 
  theme(axis.title.x = element_text(size = 15), axis.title.y = element_text(size = 15),
        plot.title = element_text(size = 16), axis.text.x=element_text(size = 12),
        axis.text.y=element_text(size = 12)) + theme(legend.position = "top") + 
  theme(legend.title = element_text(size = 12, face = "bold")) +
  theme(legend.text = element_text(size = 12))
ggsave(paste0("/Users/wdeng/Desktop/wo_int/", path, "_auc_grad.pdf"), width = 9, height = 7)


## contrast for categorical covariates
fdt_method <- filter(fdt_file, method %in% c("FDT_cat", "Extra", "RFNN"))
fdt_method[fdt_method$method == "Extra", "method"] <- "RF"
file_all <- rbind(fdt_method, R_methods)
file_all$method <- as.character(file_all$method)
tmp <- filter(file_all, (data == "matern32") & (dim == 100))
tmp$data <- ifelse(path == "cat", "mixture", ifelse(path == "cont", "continuous", path))
main_cst <- rbind(main_cst, tmp)

file <- file_all
file[file$method == "FDT_cat", "method"] <- "FDT"
file <- filter(file, method %in% c("FDT", "RFNN", "RF", "Knockoff", "BKMR", "BART", 
                                   "GAM", "BAKR", "BRR", "BL"))
file$data <- factor(file$data, order = T, levels = c("linear", "rbf", "matern32", "complex"))
file[file$method == "FDT", "method"] <- "RF-FDT (Ours)"
file[file$method == "RF", "method"] <- "RF-Impurity"
file[file$method == "BART", "method"] <- "BART"
file[file$method == "Knockoff", "method"] <- "RF-KnockOff"
file[file$method == "RFNN", "method"] <- "RFNN (Ours)"
file[file$method == "GAM", "method"] <- "GAM (Ours)"
file$method <- factor(file$method, order = T, levels = c("RF-FDT (Ours)", "RFNN (Ours)", 
                                                         "RF-Impurity", "BKMR", "BART", "BAKR", 
                                                         "RF-KnockOff",  "GAM (Ours)", "BRR", "BL"))

p2 <- ggplot(data = file, aes(x = n, y = auc, color = method, linetype = method))+ 
  scale_colour_manual(values = c("purple", "darkgreen", "blue", "darkolivegreen3", "darkslateblue", 
                                 "chartreuse4", "skyblue", "brown", "deeppink2", "brown1")) + 
  scale_linetype_manual(values = c("solid", "solid", "dashed", "dashed", "dotted", 
                                   "dotted", "dotdash", "solid", "dashed", "dotted"))+
  geom_line() + facet_grid(dim ~ data) + labs(x = "n", y = 'AUROC') + 
  geom_point(size = .5) + theme_set(theme_bw()) + ylim(0, 1) + 
  theme(axis.title.x = element_text(size = 15), axis.title.y = element_text(size = 15),
        plot.title = element_text(size = 16), axis.text.x=element_text(size = 12),
        axis.text.y=element_text(size = 12)) + theme(legend.position = "top") + 
  theme(legend.title = element_text(size = 12, face = "bold")) +
  theme(legend.text = element_text(size = 12))
ggsave(paste0("/Users/wdeng/Desktop/wo_int/", path, "_auc_cst.pdf"), width = 9, height = 7)
# 7 x 9



################################################ TST_MSE: all ################################################ 
path_name <- paste0("/Users/wdeng/Desktop/FDT/results/20220518/wo_int/", path, "/sim_mse_", path, ".txt")
file <- read.table(path_name, fill = TRUE)
names(file) <- c("data", "n", "dim", "rep", "BKMR", "MARS", "BART", "GAM", "BAKR", "BRR", "BL")
file <- filter(file, rep %in% 0:19)
file0 <- na.omit(file)
file0 <- filter(file0, dim %in% c(25, 50, 100))

col_convert <- c("BKMR", "BART", "GAM", "MARS", "BAKR", "BRR", "BL")
file0[col_convert] <- sapply(file0[col_convert], as.numeric)
file0 <- replace_with_na(file0, replace = list(GAM = -1))
file0 <- replace_with_na(file0, replace = list(BKMR = -1))

tmp <- summarise_at(group_by(file0, data, n, dim), vars(BKMR, BART, GAM, MARS, BAKR, BRR, BL), 
                    list(~mean(., na.rm = TRUE)))
tmp1 <- melt(tmp, id.vars = c("data", "n", "dim"), variable.name = "method", value.name = "tst_mse")  
tmp <- summarise_at(group_by(file0, data, n, dim), vars(BKMR, BART, GAM, MARS, BAKR, BRR, BL), 
                    list(~sd(., na.rm = TRUE)))
tmp2 <- melt(tmp, id.vars = c("data", "n", "dim"), variable.name = "method", value.name = "sd")
file0 <- merge(tmp1, tmp2)
R_methods_mse <- file0

fdt_method <- fdt_file_mse
file_all <- rbind(fdt_method, R_methods_mse)
file_all$method <- as.character(file_all$method)

file <- file_all
file$data <- factor(file$data, order = T, levels = c("linear", "rbf", "matern32", "complex"))

file <- filter(file, method %in% c("Extra", "RFNN", "BKMR", "BART",
                                   "GAM", "BAKR", "BRR", "BL"))
file[file$method == "Extra", "method"] <- "FDT"

tmp <- filter(file, (data == "matern32") & (dim == 100))
tmp$data <- ifelse(path == "cat", "mixture", ifelse(path == "cont", "continuous", path))
main_mse <- rbind(main_mse, tmp)
mse_nan$data <- ifelse(path == "cat", "mixture", ifelse(path == "cont", "continuous", path))
main_mse <- rbind(main_mse, mse_nan)

file$dim <- factor(file$dim, order = T, levels = c(25, 50, 100))
file$n <- as.numeric(file$n)

file[file$method == "FDT", "method"] <- "RF-FDT (Ours)"
file[file$method == "BART", "method"] <- "BART"
file[file$method == "RFNN", "method"] <- "RFNN (Ours)"
file[file$method == "GAM", "method"] <- "GAM (Ours)"
file$method <- factor(file$method, order = T, levels = c("RF-FDT (Ours)", "RFNN (Ours)", "BART", 
                                                         "BKMR", "GAM (Ours)", "BAKR", "BRR", "BL"))

p3 <- ggplot(data = file, aes(x = n, y = tst_mse, color = method, linetype = method))+ 
  scale_colour_manual(values = c("purple", "darkgreen", "darkslateblue", "darkolivegreen3", 
                                 "brown", "chartreuse4", "deeppink2", "brown1")) + 
  scale_linetype_manual(values = c("solid", "solid", "dotted", "dashed", "solid", 
                                   "dotted", "dashed", "dotted"))+
  scale_y_continuous(trans='log10') + 
  geom_line() + facet_grid(dim ~ data) + labs(x = "n", y = 'Testing MSE') + 
  geom_point(size = .5) + theme_set(theme_bw()) + 
  theme(axis.title.x = element_text(size = 15), axis.title.y = element_text(size = 15),
        plot.title = element_text(size = 16), axis.text.x=element_text(size = 12),
        axis.text.y=element_text(size = 12)) + theme(legend.position = "top") + 
  theme(legend.title = element_text(size = 12, face = "bold")) +
  theme(legend.text = element_text(size = 12))
ggsave(paste0("/Users/wdeng/Desktop/wo_int/", path, "_mse.pdf"), width = 9, height = 7)





################################################ AUC: main ################################################ 
dd <- filter(main_grad, data == "continuous")
main_cst <- filter(main_cst, data != "continuous")
main_cst <- rbind(main_cst, dd)
main_grad[main_grad$data == "mixture", "data"] <- "synthetic-mixture"
main_grad[main_grad$data == "continuous", "data"] <- "synthetic-continuous"
main_cst[main_cst$data == "mixture", "data"] <- "synthetic-mixture"
main_cst[main_cst$data == "continuous", "data"] <- "synthetic-continuous"
main_cst[main_cst$method == "FDT_cat", "method"] <- "FDT"
main_mse[main_mse$data == "mixture", "data"] <- "synthetic-mixture"
main_mse[main_mse$data == "continuous", "data"] <- "synthetic-continuous"
main_mse$n[153:160] <- rep(c(50, 100, 150, 257), 2)


get_legend<-function(myggplot){
  tmp <- ggplot_gtable(ggplot_build(myggplot))
  leg <- which(sapply(tmp$grobs, function(x) x$name) == "guide-box")
  legend <- tmp$grobs[[leg]]
  return(legend)
}

main_cst$type <- "AUC"
main_mse$type <- "Testing MSE"
names(main_cst)[5] <- "value"
names(main_mse)[5] <- "value"

file <- rbind(main_cst, main_mse)
file$n <- as.integer(file$n)
file <- filter(file, method %in% c("FDT", "RFNN", "RF", "Knockoff", "BKMR", "BART", 
                                   "GAM", "BAKR", "BRR", "BL"))
file[file$method == "FDT", "method"] <- "RF-FDT (Ours)"
file[file$method == "RF", "method"] <- "RF-Impurity"
file[file$method == "BART", "method"] <- "BART"
file[file$method == "Knockoff", "method"] <- "RF-KnockOff"
file[file$method == "RFNN", "method"] <- "RFNN (Ours)"
file[file$method == "GAM", "method"] <- "GAM (Ours)"

file$method <- factor(file$method, order = T, levels = c("RF-FDT (Ours)", "RFNN (Ours)", 
                                                         "RF-Impurity", "BKMR", "BART", "BAKR", 
                                                         "RF-KnockOff",  "GAM (Ours)", "BRR", "BL"))

file$data <- factor(file$data, order = T, levels = c("synthetic-mixture", "synthetic-continuous", 
                                                     "adult", "heart", "mi"))

fdt0 <- ggplot(filter(file, (data == "synthetic-mixture") & (type == "Testing MSE")), 
               aes(x = n, y = value, color = method, linetype = method))+ 
  scale_colour_manual(values = c("purple", "darkgreen", "blue", "darkolivegreen3", "darkslateblue", 
                                 "chartreuse4", "skyblue", "brown", "deeppink2", "brown1")) + 
  scale_linetype_manual(values = c("solid", "solid", "dashed", "dashed", "dotted", 
                                   "dotted", "dotdash", "solid", "dashed", "dotted"))+
  scale_y_continuous(trans='log10') + 
  geom_line() + labs(x = "n", y = "Testing MSE") + 
  geom_point(size = .2) + theme_set(theme_bw()) + ggtitle("FDT") + 
  theme(plot.title = element_text(hjust = 0.5), 
        axis.title.x=element_blank(), axis.text.x=element_blank(), axis.ticks.x=element_blank(), 
        axis.title.y = element_blank(), legend.position = "top", legend.title = element_text(size=12), 
        legend.text = element_text(size=12))
legend <- get_legend(fdt0)


q1 <- ggplot(filter(file, (data == "synthetic-mixture") & (type == "Testing MSE")), 
               aes(x = n, y = value, color = method, linetype = method))+ 
  scale_colour_manual(values = c("purple", "darkgreen", "blue", "darkolivegreen3", "darkslateblue", 
                                 "chartreuse4", "skyblue", "brown", "deeppink2", "brown1")) + 
  scale_linetype_manual(values = c("solid", "solid", "dashed", "dashed", "dotted", 
                                   "dotted", "dotdash", "solid", "dashed", "dotted"))+
  ylim(0, 3) + 
  geom_line() + labs(x = "n", y = "Testing MSE") + 
  geom_point(size = .2) + theme_set(theme_bw()) + 
  theme(axis.title.x=element_blank()) + theme(legend.position = "none")

q2 <- ggplot(filter(file, (data == "synthetic-continuous") & (type == "Testing MSE")), 
             aes(x = n, y = value, color = method, linetype = method))+ 
  scale_colour_manual(values = c("purple", "darkgreen", "blue", "darkolivegreen3", "darkslateblue", 
                                 "chartreuse4", "skyblue", "brown", "deeppink2", "brown1")) + 
  scale_linetype_manual(values = c("solid", "solid", "dashed", "dashed", "dotted", 
                                   "dotted", "dotdash", "solid", "dashed", "dotted"))+
  ylim(0, 3) + 
  geom_line() + labs(x = "n", y = "Testing MSE") + 
  geom_point(size = .2) + theme_set(theme_bw()) + 
  theme(axis.title.x=element_blank(), axis.title.y=element_blank()) + theme(legend.position = "none")


q3 <- ggplot(filter(file, (data == "adult") & (type == "Testing MSE")), 
             aes(x = n, y = value, color = method, linetype = method))+ 
  scale_colour_manual(values = c("purple", "darkgreen", "blue", "darkolivegreen3", "darkslateblue", 
                                 "chartreuse4", "skyblue", "brown", "deeppink2", "brown1")) + 
  scale_linetype_manual(values = c("solid", "solid", "dashed", "dashed", "dotted", 
                                   "dotted", "dotdash", "solid", "dashed", "dotted"))+
  ylim(0, 3) + 
  geom_line() + labs(x = "n", y = "Testing MSE") + 
  geom_point(size = .2) + theme_set(theme_bw()) + 
  theme(axis.title.x=element_blank(), axis.title.y=element_blank()) + theme(legend.position = "none")


q4 <- ggplot(filter(file, (data == "heart") & (type == "Testing MSE")), 
             aes(x = n, y = value, color = method, linetype = method))+ 
  scale_colour_manual(values = c("purple", "darkgreen", "blue", "darkolivegreen3", "darkslateblue", 
                                 "chartreuse4", "skyblue", "brown", "deeppink2", "brown1")) + 
  scale_linetype_manual(values = c("solid", "solid", "dashed", "dashed", "dotted", 
                                   "dotted", "dotdash", "solid", "dashed", "dotted"))+
  ylim(0, 3) + 
  geom_line() + labs(x = "n", y = "Testing MSE") + 
  geom_point(size = .2) + theme_set(theme_bw()) + 
  theme(axis.title.x=element_blank(), axis.title.y=element_blank()) + theme(legend.position = "none")


q5 <- ggplot(filter(file, (data == "mi") & (type == "Testing MSE")), 
             aes(x = n, y = value, color = method, linetype = method))+ 
  scale_colour_manual(values = c("purple", "darkgreen", "blue", "darkolivegreen3", "darkslateblue", 
                                 "chartreuse4", "skyblue", "brown", "deeppink2", "brown1")) + 
  scale_linetype_manual(values = c("solid", "solid", "dashed", "dashed", "dotted", 
                                   "dotted", "dotdash", "solid", "dashed", "dotted"))+
  ylim(0, 3) + 
  geom_line() + labs(x = "n", y = "Testing MSE") + 
  geom_point(size = .2) + theme_set(theme_bw()) + 
  theme(axis.title.x=element_blank(), axis.title.y=element_blank()) + theme(legend.position = "none")


q6 <- ggplot(filter(file, (data == "synthetic-mixture") & (type == "AUC")), 
             aes(x = n, y = value, color = method, linetype = method))+ 
  scale_colour_manual(values = c("purple", "darkgreen", "blue", "darkolivegreen3", "darkslateblue", 
                                 "chartreuse4", "skyblue", "brown", "deeppink2", "brown1")) + 
  scale_linetype_manual(values = c("solid", "solid", "dashed", "dashed", "dotted", 
                                   "dotted", "dotdash", "solid", "dashed", "dotted"))+
  geom_line() + labs(x = "n", y = "AUROC") + ggtitle("synthetic-mixture") + 
  geom_point(size = .2) + theme_set(theme_bw()) + 
  theme(plot.title = element_text(hjust = 0.5, size = 12, face = "bold")) +
  theme(axis.title.x=element_blank()) + theme(legend.position = "none")


q7 <- ggplot(filter(file, (data == "synthetic-continuous") & (type == "AUC")), 
             aes(x = n, y = value, color = method, linetype = method))+ 
  scale_colour_manual(values = c("purple", "darkgreen", "blue", "darkolivegreen3", "darkslateblue", 
                                 "chartreuse4", "skyblue", "brown", "deeppink2", "brown1")) + 
  scale_linetype_manual(values = c("solid", "solid", "dashed", "dashed", "dotted", 
                                   "dotted", "dotdash", "solid", "dashed", "dotted"))+
  geom_line() + labs(x = "n", y = "AUROC") + ggtitle("synthetic-continuous") + 
  geom_point(size = .2) + theme_set(theme_bw()) + 
  theme(plot.title = element_text(hjust = 0.5, size = 12, face = "bold")) +
  theme(axis.title.x=element_blank(), axis.title.y=element_blank()) + theme(legend.position = "none")


q8 <- ggplot(filter(file, (data == "adult") & (type == "AUC")), 
             aes(x = n, y = value, color = method, linetype = method))+ 
  scale_colour_manual(values = c("purple", "darkgreen", "blue", "darkolivegreen3", "darkslateblue", 
                                 "chartreuse4", "skyblue", "brown", "deeppink2", "brown1")) + 
  scale_linetype_manual(values = c("solid", "solid", "dashed", "dashed", "dotted", 
                                   "dotted", "dotdash", "solid", "dashed", "dotted"))+
  geom_line() + labs(x = "n", y = "AUROC") + ggtitle("adult") + 
  geom_point(size = .2) + theme_set(theme_bw()) + 
  theme(plot.title = element_text(hjust = 0.5, size = 12, face = "bold")) +
  theme(axis.title.x=element_blank(), axis.title.y=element_blank()) + theme(legend.position = "none")


q9 <- ggplot(filter(file, (data == "heart") & (type == "AUC")), 
             aes(x = n, y = value, color = method, linetype = method))+ 
  scale_colour_manual(values = c("purple", "darkgreen", "blue", "darkolivegreen3", "darkslateblue", 
                                 "chartreuse4", "skyblue", "brown", "deeppink2", "brown1")) + 
  scale_linetype_manual(values = c("solid", "solid", "dashed", "dashed", "dotted", 
                                   "dotted", "dotdash", "solid", "dashed", "dotted"))+
  geom_line() + labs(x = "n", y = "AUROC") + ggtitle("heart") + 
  geom_point(size = .2) + theme_set(theme_bw()) + 
  theme(plot.title = element_text(hjust = 0.5, size = 12, face = "bold")) +
  theme(axis.title.x=element_blank(), axis.title.y=element_blank()) + theme(legend.position = "none")

q10 <- ggplot(filter(file, (data == "mi") & (type == "AUC")), 
             aes(x = n, y = value, color = method, linetype = method))+ 
  scale_colour_manual(values = c("purple", "darkgreen", "blue", "darkolivegreen3", "darkslateblue", 
                                 "chartreuse4", "skyblue", "brown", "deeppink2", "brown1")) + 
  scale_linetype_manual(values = c("solid", "solid", "dashed", "dashed", "dotted", 
                                   "dotted", "dotdash", "solid", "dashed", "dotted"))+
  geom_line() + labs(x = "n", y = "AUROC") + ggtitle("mi") + 
  geom_point(size = .2) + theme_set(theme_bw()) + 
  theme(plot.title = element_text(hjust = 0.5, size = 12, face = "bold")) +
  theme(axis.title.x=element_blank(), axis.title.y=element_blank()) + theme(legend.position = "none")



grid.arrange(legend, q6, q7, q8, q9, q10, q1, q2, q3, q4, q5, ncol=5, nrow=3, 
             layout_matrix = rbind(c(1,1,1,1,1), c(2,3,4,5,6), c(7,8,9,10,11)),
             widths = c(3.2, 3, 3, 3, 3), heights = c(1.0, 2.5, 2.3), bottom="n")


# 6 x 13
