setwd('/Users/wdeng/Desktop/FDT/real_data/bangladesh/vi')
library(ggplot2)
library(dplyr)
library(reshape2)
library(MASS)
library(gridExtra)

get_legend<-function(myggplot){
  tmp <- ggplot_gtable(ggplot_build(myggplot))
  leg <- which(sapply(tmp$grobs, function(x) x$name) == "guide-box")
  legend <- tmp$grobs[[leg]]
  return(legend)
}

file <- read.csv("fdt_vi_c1000.csv", header = TRUE)
file <- file / max(file)
sorted_unique <- sort(unique(as.vector(as.matrix(file))))
prop_tmp <- sapply(1:length(sorted_unique), function(j) colMeans(file > sorted_unique[j]))
prop_matrix <- as.data.frame(cbind(threshold = sorted_unique, t(prop_tmp)))
prop_file <- melt(prop_matrix, id.vars = "threshold", variable.name = "covariate", value.name = "cdf")

# plot all covariates
fdt0 <- ggplot(prop_file, aes(x = threshold, y = cdf, color = covariate))+ 
  geom_line() + labs(x = "s", y = expression(paste("P(", psi, ">s)"))) + 
  geom_point(size = .2) + theme_set(theme_bw()) + ggtitle("FDT") + 
  theme(plot.title = element_text(hjust = 0.5), 
        axis.title.x=element_blank(), axis.text.x=element_blank(), axis.ticks.x=element_blank(), 
        axis.title.y = element_blank(), legend.position = "top", legend.title = element_text(size=12), 
        legend.text = element_text(size=12))
legend <- get_legend(fdt0)

fdt <- ggplot(prop_file, aes(x = threshold, y = cdf, color = covariate))+ 
  geom_line() + labs(x = "s", y = expression(paste("P(", psi, ">s)"))) + 
  geom_point(size = .2) + theme_set(theme_bw()) + ggtitle("FDT") + 
  theme(plot.title = element_text(hjust = 0.5, size = 10)) +
  theme(axis.title.x=element_blank(), axis.text.x=element_blank(), axis.ticks.x=element_blank(), 
        axis.title.y = element_blank()) + 
  theme(legend.position = "none")
# 5 x 7.5

file <- read.csv("gam_beta2.csv", header = TRUE)
file <- file / max(file)
sorted_unique <- sort(unique(as.vector(as.matrix(file))))
prop_tmp <- sapply(1:length(sorted_unique), function(j) colMeans(file > sorted_unique[j]))
prop_matrix <- as.data.frame(cbind(threshold = sorted_unique, t(prop_tmp)))
prop_file <- melt(prop_matrix, id.vars = "threshold", variable.name = "covariate", value.name = "cdf")

# plot all covariates
gam <- ggplot(prop_file, aes(x = threshold, y = cdf, color = covariate))+ 
  geom_line() + labs(x = "s", y = expression(paste("P(", psi, ">s)"))) + 
  geom_point(size = .2) + theme_set(theme_bw()) + ggtitle("GAM") + 
  theme(plot.title = element_text(hjust = 0.5, size = 10)) +
  theme(axis.title.x=element_blank(), axis.text.x=element_blank(), axis.ticks.x=element_blank(), 
        axis.title.y=element_blank(), axis.text.y=element_blank(), axis.ticks.y=element_blank()) +
  theme(legend.position="none")


file <- read.csv("brr_beta2.csv", header = TRUE)
file <- file / max(file)
sorted_unique <- sort(unique(as.vector(as.matrix(file))))
prop_tmp <- sapply(1:length(sorted_unique), function(j) colMeans(file > sorted_unique[j]))
prop_matrix <- as.data.frame(cbind(threshold = sorted_unique, t(prop_tmp)))
prop_file <- melt(prop_matrix, id.vars = "threshold", variable.name = "covariate", value.name = "cdf")

# plot all covariates
brr <- ggplot(prop_file, aes(x = threshold, y = cdf, color = covariate))+ 
  geom_line() + labs(x = "s", y = expression(paste("P(", psi, ">s)"))) + 
  geom_point(size = .2) + theme_set(theme_bw()) + ggtitle("BRR") + 
  theme(plot.title = element_text(hjust = 0.5, size = 10)) +
  theme(axis.title.x=element_blank(), axis.text.x=element_blank(), axis.ticks.x=element_blank(), 
        axis.title.y=element_blank(), axis.text.y=element_blank(), axis.ticks.y=element_blank()) +
  theme(legend.position="none")


file <- read.csv("rf_vi.csv", header = TRUE)
file <- file / max(file)
sorted_unique <- sort(unique(as.vector(as.matrix(file))))
prop_tmp <- sapply(1:length(sorted_unique), function(j) colMeans(file > sorted_unique[j]))
prop_matrix <- as.data.frame(cbind(threshold = sorted_unique, t(prop_tmp)))
prop_file <- melt(prop_matrix, id.vars = "threshold", variable.name = "covariate", value.name = "cdf")

# plot all covariates
rf <- ggplot(prop_file, aes(x = threshold, y = cdf, color = covariate))+ 
  geom_line() + labs(x = "s", y = expression(paste("P(", psi, ">s)"))) + 
  geom_point(size = .2) + theme_set(theme_bw()) + ggtitle("RF") + 
  theme(plot.title = element_text(hjust = 0.5, size = 10)) +
  theme(axis.title.y = element_blank(), axis.title.x = element_blank()) + 
  theme(legend.position = "none") 


file <- read.csv("bakr_beta2.csv", header = TRUE)
set.seed(0921)
file <- file[sample.int(nrow(file), size=50), ]
file <- file / max(file)
sorted_unique <- sort(unique(as.vector(as.matrix(file))))
prop_tmp <- sapply(1:length(sorted_unique), function(j) colMeans(file > sorted_unique[j]))
prop_matrix <- as.data.frame(cbind(threshold = sorted_unique, t(prop_tmp)))
prop_file <- melt(prop_matrix, id.vars = "threshold", variable.name = "covariate", value.name = "cdf")

# plot all covariates
bakr <- ggplot(prop_file, aes(x = threshold, y = cdf, color = covariate))+ 
  geom_line() + labs(x = "s", y = expression(paste("P(", psi, ">s)"))) + 
  geom_point(size = .2) + theme_set(theme_bw()) + ggtitle("BAKR") + 
  theme(plot.title = element_text(hjust = 0.5, size = 10)) +
  theme(axis.title.y=element_blank(), axis.text.y=element_blank(), 
        axis.ticks.y=element_blank(), axis.title.x = element_blank()) +
  theme(legend.position="none")


file <- read.csv("bl_beta2.csv", header = TRUE)
file <- file / max(file)
sorted_unique <- sort(unique(as.vector(as.matrix(file))))
prop_tmp <- sapply(1:length(sorted_unique), function(j) colMeans(file > sorted_unique[j]))
prop_matrix <- as.data.frame(cbind(threshold = sorted_unique, t(prop_tmp)))
prop_file <- melt(prop_matrix, id.vars = "threshold", variable.name = "covariate", value.name = "cdf")

# plot all covariates
bl <- ggplot(prop_file, aes(x = threshold, y = cdf, color = covariate))+ 
  geom_line() + labs(x = "s", y = expression(paste("P(", psi, ">s)"))) + 
  geom_point(size = .2) + theme_set(theme_bw()) + ggtitle("BL") + 
  theme(plot.title = element_text(hjust = 0.5, size = 10)) +
  theme(axis.title.y=element_blank(), axis.text.y=element_blank(), 
        axis.ticks.y=element_blank(), axis.title.x = element_blank()) +
  theme(legend.position="none")

# left <- textGrob("Inclusion Probability", rot = 90, gp = gpar(fontsize = 17))
# bottom <- textGrob("Effect Magnitude Threshold", gp = gpar(fontsize = 15))

grid.arrange(legend, fdt, gam, brr, rf, bakr, bl,  ncol=3, nrow=3, 
             layout_matrix = rbind(c(1,1,1), c(2,3,4), c(5,6,7)),
             widths = c(3.2, 2.7, 2.7), heights = c(1.0, 2.5, 2.7), 
             left="P(psi>s)", bottom="s")







fdt <- ggplot(prop_file, aes(x = threshold, y = cdf, color = covariate))+ 
  geom_line() + labs(x = "s", y = expression(paste("P(", psi, ">s)"))) + 
  geom_point(size = .2) + theme_set(theme_bw()) + ggtitle("FDT") + 
  theme(plot.title = element_text(hjust = 0.5, size = 10)) +
  theme(axis.title.x=element_blank(), axis.title.y = element_blank()) + 
  theme(legend.position = "none")
# 5 x 7.5

file <- read.csv("gam_beta2.csv", header = TRUE)
file <- file / max(file)
sorted_unique <- sort(unique(as.vector(as.matrix(file))))
prop_tmp <- sapply(1:length(sorted_unique), function(j) colMeans(file > sorted_unique[j]))
prop_matrix <- as.data.frame(cbind(threshold = sorted_unique, t(prop_tmp)))
prop_file <- melt(prop_matrix, id.vars = "threshold", variable.name = "covariate", value.name = "cdf")

# plot all covariates
gam <- ggplot(prop_file, aes(x = threshold, y = cdf, color = covariate))+ 
  geom_line() + labs(x = "s", y = expression(paste("P(", psi, ">s)"))) + 
  geom_point(size = .2) + theme_set(theme_bw()) + ggtitle("GAM") + 
  theme(plot.title = element_text(hjust = 0.5, size = 10)) +
  theme(axis.title.x=element_blank(), 
        axis.title.y=element_blank(), axis.text.y=element_blank(), axis.ticks.y=element_blank()) +
  theme(legend.position="none")


file <- read.csv("brr_beta2.csv", header = TRUE)
file <- file / max(file)
sorted_unique <- sort(unique(as.vector(as.matrix(file))))
prop_tmp <- sapply(1:length(sorted_unique), function(j) colMeans(file > sorted_unique[j]))
prop_matrix <- as.data.frame(cbind(threshold = sorted_unique, t(prop_tmp)))
prop_file <- melt(prop_matrix, id.vars = "threshold", variable.name = "covariate", value.name = "cdf")

# plot all covariates
brr <- ggplot(prop_file, aes(x = threshold, y = cdf, color = covariate))+ 
  geom_line() + labs(x = "s", y = expression(paste("P(", psi, ">s)"))) + 
  geom_point(size = .2) + theme_set(theme_bw()) + ggtitle("BRR") + 
  theme(plot.title = element_text(hjust = 0.5, size = 10)) +
  theme(axis.title.x=element_blank(), axis.title.y = element_blank()) + 
  theme(legend.position = "none")


file <- read.csv("rf_vi.csv", header = TRUE)
file <- file / max(file)
sorted_unique <- sort(unique(as.vector(as.matrix(file))))
prop_tmp <- sapply(1:length(sorted_unique), function(j) colMeans(file > sorted_unique[j]))
prop_matrix <- as.data.frame(cbind(threshold = sorted_unique, t(prop_tmp)))
prop_file <- melt(prop_matrix, id.vars = "threshold", variable.name = "covariate", value.name = "cdf")

# plot all covariates
rf <- ggplot(prop_file, aes(x = threshold, y = cdf, color = covariate))+ 
  geom_line() + labs(x = "s", y = expression(paste("P(", psi, ">s)"))) + 
  geom_point(size = .2) + theme_set(theme_bw()) + ggtitle("RF") + 
  theme(plot.title = element_text(hjust = 0.5, size = 10)) +
  theme(axis.title.y = element_blank(), axis.title.x = element_blank()) + 
  theme(legend.position = "none") 


file <- read.csv("bakr_beta2.csv", header = TRUE)
set.seed(0921)
file <- file[sample.int(nrow(file), size=50), ]
file <- file / max(file)
sorted_unique <- sort(unique(as.vector(as.matrix(file))))
prop_tmp <- sapply(1:length(sorted_unique), function(j) colMeans(file > sorted_unique[j]))
prop_matrix <- as.data.frame(cbind(threshold = sorted_unique, t(prop_tmp)))
prop_file <- melt(prop_matrix, id.vars = "threshold", variable.name = "covariate", value.name = "cdf")

# plot all covariates
bakr <- ggplot(prop_file, aes(x = threshold, y = cdf, color = covariate))+ 
  geom_line() + labs(x = "s", y = expression(paste("P(", psi, ">s)"))) + 
  geom_point(size = .2) + theme_set(theme_bw()) + ggtitle("BAKR") + 
  theme(plot.title = element_text(hjust = 0.5, size = 10)) +
  theme(axis.title.y=element_blank(), axis.text.y=element_blank(), 
        axis.ticks.y=element_blank(), axis.title.x = element_blank()) +
  theme(legend.position="none")


file <- read.csv("bl_beta2.csv", header = TRUE)
file <- file / max(file)
sorted_unique <- sort(unique(as.vector(as.matrix(file))))
prop_tmp <- sapply(1:length(sorted_unique), function(j) colMeans(file > sorted_unique[j]))
prop_matrix <- as.data.frame(cbind(threshold = sorted_unique, t(prop_tmp)))
prop_file <- melt(prop_matrix, id.vars = "threshold", variable.name = "covariate", value.name = "cdf")

# plot all covariates
bl <- ggplot(prop_file, aes(x = threshold, y = cdf, color = covariate))+ 
  geom_line() + labs(x = "s", y = expression(paste("P(", psi, ">s)"))) + 
  geom_point(size = .2) + theme_set(theme_bw()) + ggtitle("BL") + 
  theme(plot.title = element_text(hjust = 0.5, size = 10)) +
  theme(axis.title.y=element_blank(), axis.text.y=element_blank(), 
        axis.ticks.y=element_blank(), axis.title.x = element_blank()) +
  theme(legend.position="none")


empty_plot <- ggplot() + theme_void()
grid.arrange(legend, fdt, gam, bakr, brr, bl, empty_plot, ncol=3, nrow=3, 
             layout_matrix = rbind(c(1,1,1), c(2,3,4), c(5,6,7)),
             widths = c(3.2, 2.7, 2.7), heights = c(1.0, 2.5, 2.7), 
             left="P(psi>s)", bottom="s")

# 6 x 9

# plot the first ten covariates
ten_cov <- paste0("x", 1:10)
prop_ten <- filter(prop_file, covariate %in% ten_cov)
ggplot(prop_ten, aes(x = threshold, y = cdf, color = covariate))+ 
  geom_line() + labs(x = "s", y = expression(paste("P(", psi, ">s)"))) + 
  geom_point(size = .2) + theme_set(theme_bw()) + 
  theme(axis.title.x = element_text(size = 15), axis.title.y = element_text(size = 15),
        plot.title = element_text(size = 16), axis.text.x=element_text(size = 12),
        axis.text.y=element_text(size = 12)) + theme(legend.position = "top") + 
  theme(legend.title = element_text(size = 12, face = "bold")) +
  theme(legend.text = element_text(size = 12))



# data preparation
library(gamsel)
library(mvtnorm)

data <- read.csv("/Users/wdeng/Downloads/real_data/bangladesh/data_processed.csv")
# bangladesh data[, c(1, 2, 4, 35:42)]
x_train <- data[1:665, 2:11]
y_train <- data[1:665, 1]  
x_test <- data[666:715, 2:11]
y_test <- data[666:715, 1]  

mcmc.iter <- 2e3
mcmc.burn <- 1e3
ETA <- list(list(X = x_train, model = "BRR"))
reg.BRR <- BGLR(y = y_train, ETA = ETA, nIter = mcmc.iter,
                burnIn = mcmc.burn, verbose = FALSE)
beta_BRR <- rmvnorm(50, mean = reg.BRR$ETA[[1]]$b, sigma = diag(reg.BRR$ETA[[1]]$SD.b^2))
beta_BRR2 <- beta_BRR^2
beta_BRR2 <- as.data.frame(beta_BRR2)
names(beta_BRR2) <- c('clinic', 'sex', 'prot', 'fat', 'carb', 'fib', 
                      'ash', 'as_ln', 'mn_ln', 'pb_ln')
write.csv(beta_BRR2, "/Users/irisdeng/Downloads/concrete/vi/brr_beta2.csv", row.names = FALSE)


ETA <- list(list(X = x_train, model = "BL"))
reg.BL <- BGLR(y = y_train, ETA = ETA, nIter = mcmc.iter,
               burnIn = mcmc.burn, verbose = FALSE)
beta_BL <- rmvnorm(50, mean = reg.BL$ETA[[1]]$b, sigma = diag(reg.BL$ETA[[1]]$SD.b^2))
beta_BL2 <- beta_BL^2
beta_BL2 <- as.data.frame(beta_BL2)
names(beta_BL2) <- c('clinic', 'sex', 'prot', 'fat', 'carb', 'fib', 
                     'ash', 'as_ln', 'mn_ln', 'pb_ln')
write.csv(beta_BL2, "/Users/irisdeng/Downloads/concrete/vi/bl_beta2.csv", row.names = FALSE)


fit <- gamsel(x_train, y_train, degrees = rep(10, 8))
beta_gam <- t(fit$alphas)
beta_gam2 <- beta_gam^2
beta_gam2 <- as.data.frame(beta_gam2)
names(beta_gam2) <- c('clinic', 'sex', 'prot', 'fat', 'carb', 'fib', 
                      'ash', 'as_ln', 'mn_ln', 'pb_ln')
write.csv(beta_gam2, "/Users/irisdeng/Downloads/concrete/vi/gam_beta2.csv", row.names = FALSE)





