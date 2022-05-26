
md_comp <- function(name = "linear", n = 100, dim = 25, i = 0, data_name = "cat") {
  
  data_name <- as.character(data_name)
  filename <- paste0("data=", name, "_n=", n, "_dim=", dim, "_i=", i)
  data <- read.csv(paste0("/n/home10/irisdeng/FDT/python/experiments/expr/datasets/", data_name, "/",
                          name, "_n", n, "_d", dim, "_i", i, ".csv"), row.names = 1)
  df_train <- data[1:n, ]
  x_train <- df_train[, -c(1, 2)]
  y_train <- df_train[, 2]
  # f_train <- df_train[, 1]
  df_train <- df_train[, -1]
  
  df_test <- data[(n+1):nrow(data), ]
  x_test <- df_test[, -c(1, 2)]
  y_test <- df_test[, 2]
  # f_test <- df_test[, 1]
  df_test <- df_test[, -1]
  
  true <- c(rep(1, 5), rep(0, dim - 5))
  
  # # random forest
  # start_time <- Sys.time()
  # res_rf <- randomForest(y ~ ., data = df_train, importance = TRUE)
  # end_time <- Sys.time()
  # rf_time <- difftime(end_time, start_time, units = "secs")[[1]]
  # pred <- predict(res_rf, x_test)
  # rf_mse <- mean((y_test - pred) ^ 2)
  # # rf_mse <- mean((pred - f_test)^2) / mean((f_test - mean(f_test))^2)
  # rf_est <- res_rf$importance[, 2]
  # rf_est <- rf_est / max(rf_est)
  # rf_auc <- auc(roc(true, rf_est))
  # 
  # knockoff
  start_time <- Sys.time()
  knock_rf <- knockoff.filter(x_train, y_train, statistic = stat.random_forest)
  end_time <- Sys.time()
  knock_time <- difftime(end_time, start_time, units = "secs")[[1]]
  knock_est <- knock_rf$statistic
  knock_est <- knock_est - min(knock_est)
  knock_est <- knock_est / max(knock_est)
  knock_auc <- auc(roc(true, knock_est))

  # bnn
  start_time <- Sys.time()

  bnn_tmp <- tryCatch(bnn_res <- BNNsel(x_train, y_train, train_num = as.integer(0.95 * length(y_train)),
                        total_iteration = 5000),
             error = function(e) e)

  if ("error" %in% class(bnn_tmp)) {
    bnn_time <- -1
    bnn_auc <- -1
  } else {
    end_time <- Sys.time()
    bnn_time <- difftime(end_time, start_time, units = "secs")[[1]]
    bnn_tmp2 <- tryCatch(bnn_auc <- auc(roc(true, bnn_res$mar[-1])), error = function(e) e)
    # length doesn't match sometimes
    if ("error" %in% class(bnn_tmp2)) {
      bnn_auc <- -1
    } else {
      bnn_auc <- auc(roc(true, bnn_res$mar[-1]))
    }
  }

  # bkmr
  start_time <- Sys.time()
  fitkm <- kmbayes(y = y_train, Z = x_train, iter = 4000, verbose = FALSE, varsel = TRUE)
  end_time <- Sys.time()
  bkmr_time <- difftime(end_time, start_time, units = "secs")[[1]]
  res_bkmr <- ExtractPIPs(fitkm)
  bkmr_auc <- auc(roc(true, res_bkmr$PIP))
  bkmr_tmp <- tryCatch(pred <- SamplePred(fitkm, Znew = x_test,  Xnew = cbind(0)), 
                       error = function(e) e)
  # la.svd(x, nu, nv) : error code 1 from lapack routine 'dgesdd'
  if ("error" %in% class(bkmr_tmp)) {
    bkmr_mse <- -1
  } else {
    diff_pred <- y_test - pred
    diff_trunc <- diff_pred[abs(diff_pred) < 3]
    bkmr_mse <- mean(diff_trunc ^ 2)
  }
  
  # bart
  start_time <- Sys.time()
  bart_machine <- bartMachine(x_train, y_train, replace_missing_data_with_x_j_bar = TRUE)
  vs <- var_selection_by_permute(bart_machine, num_permute_samples = 100, plot = FALSE)
  end_time <- Sys.time()
  bart_time <- difftime(end_time, start_time, units = "secs")[[1]]
  pred <- predict(bart_machine, new_data = x_test)
  diff_pred <- y_test - pred
  diff_trunc <- diff_pred[abs(diff_pred) < 3]
  bart_mse <- mean(diff_trunc ^ 2)
  bart_pip <- vs$var_true_props_avg[ord_varimp(vs$var_true_props_avg)]
  bart_pip <- bart_pip / max(bart_pip)
  bart_auc <- auc(roc(true, bart_pip))
  
  # bstarss
  xnam <- paste("x", 1:dim, sep="")
  fmla <- as.formula(paste("y ~ ", paste(xnam, collapse= "+")))
  # start_time <- Sys.time()
  # m <- spikeSlabGAM(formula = fmla, data = df_train)
  # end_time <- Sys.time()
  # spike_time <- difftime(end_time, start_time, units = "secs")[[1]]
  # pred <- predict(m, newdata = x_test)
  # spike_mse <- mean((y_test - pred) ^ 2)
  # spike_pip <- summary(m)$trmSummary[, 2][-1]
  # spike_pip <- spike_pip / max(spike_pip)
  # spike_auc <- auc(roc(true, spike_pip))
  
  # gam
  if (n > dim) {
    start_time <- Sys.time()
    b <- gam(formula = fmla, data = df_train, select = TRUE, method = "REML")
    end_time <- Sys.time()
    gam_time <- difftime(end_time, start_time, units = "secs")[[1]]
    pred <- predict(b, newdata = df_test[, -1])
    diff_pred <- y_test - pred
    diff_trunc <- diff_pred[abs(diff_pred) < 3]
    gam_mse <- mean(diff_trunc ^ 2)
    gam_coef <- abs(summary(b)$p.coeff[-1])
    gam_pip <- gam_coef / max(gam_coef)
    gam_auc <- auc(roc(true, gam_pip))
  } else {
    gam_time <- -1
    gam_mse <- -1
    gam_pip <- -1
    gam_auc <- -1
  }

  # mars
  start_time <- Sys.time()
  earth_mod <- earth(formula = fmla, data = df_train)
  end_time <- Sys.time()
  earth_time <- difftime(end_time, start_time, units = "secs")[[1]]
  pred <- predict(earth_mod, newdata = df_test[, -1])
  diff_pred <- y_test - pred
  diff_trunc <- diff_pred[abs(diff_pred) < 3]
  earth_mse <- mean(diff_trunc ^ 2)
  ev <- evimp(earth_mod, trim = FALSE)
  earth_gcv <- ev[, 4]
  earth_pip <- earth_gcv / max(earth_gcv)
  earth_auc <- auc(roc(true, earth_pip))
  # 
  # bakr
  start_time <- Sys.time()
  Kn <- ApproxGaussKernel(t(x_train), dim, dim)
  v <- matrix(1, n, 1)
  M <- diag(n) - v %*% t(v) / n
  Kn <- M %*% Kn %*% M
  Kn <- Kn / mean(diag(Kn))
  evd <- EigDecomp(Kn)
  explained_var <- cumsum(evd$lambda / sum(evd$lambda))
  q <- 1:min(which(explained_var >= 0.99))
  Lambda <- diag(sort(evd$lambda, decreasing = TRUE)[q]^(-1))
  U <- evd$U[, q]
  B <- InverseMap(t(x_train), U)
  mcmc.iter <- 2e3
  mcmc.burn <- 1e3
  Gibbs <- BAKRGibbs(U, y_train, Lambda, mcmc.iter, mcmc.burn)
  theta.out <- PostMean(Gibbs$theta)
  beta.out <- PostBeta(B, theta.out)
  end_time <- Sys.time()
  bakr_time <- difftime(end_time, start_time, units = "secs")[[1]]
  names(beta.out) <- colnames(x_train)
  pred <- as.matrix(x_test) %*% beta.out
  diff_pred <- y_test - pred
  diff_trunc <- diff_pred[abs(diff_pred) < 3]
  bakr_mse <- mean(diff_trunc ^ 2)
  # bakr_mse <- mean((pred - f_test)^2) / mean((f_test - mean(f_test))^2)
  psi_bakr <- beta.out^2
  psi_bakr <- psi_bakr / max(psi_bakr)
  bakr_auc <- auc(roc(true, psi_bakr[, 1]))

  # Bayesian Ridge Regression
  start_time <- Sys.time()
  ETA <- list(list(X = x_train, model = "BRR"))
  reg.BRR <- BGLR(y = y_train, ETA = ETA, nIter = mcmc.iter,
                  burnIn = mcmc.burn, verbose = FALSE)
  end_time <- Sys.time()
  brr_time <- difftime(end_time, start_time, units = "secs")[[1]]
  brr_beta <- reg.BRR$ETA[[1]]$b
  pred <- as.matrix(x_test) %*% as.matrix(brr_beta)
  diff_pred <- y_test - pred
  diff_trunc <- diff_pred[abs(diff_pred) < 3]
  brr_mse <- mean(diff_trunc ^ 2)
  # brr_mse <- mean((pred - f_test)^2) / mean((f_test - mean(f_test))^2)
  psi_brr <- brr_beta^2
  psi_brr <- psi_brr / max(psi_brr)
  brr_auc <- auc(roc(true, psi_brr))

  # Bayesian LMM
  # start_time <- Sys.time()
  # K <- x_train %*% t(x_train)
  # ETA <- list(list(K = K, model = "RKHS"))
  # reg.BBLUP <- BGLR(y = y_train, ETA = ETA, nIter = mcmc.iter,
  #                   burnIn = mcmc.burn, verbose = FALSE)
  # reg.BBLUP_b <- ginv(x_train) %*% reg.BBLUP$ETA[[1]]$u
  # end_time <- Sys.time()
  # bblup_time <- difftime(end_time, start_time, units = "secs")[[1]]
  # pred <- as.matrix(x_test) %*% reg.BBLUP_b
  # bblup_mse <- mean((y_test - pred) ^ 2)
  # # bblup_mse <- mean((pred - f_test)^2) / mean((f_test - mean(f_test))^2)
  # psi_bblup <- reg.BBLUP_b^2
  # psi_bblup <- psi_bblup / max(psi_bblup)
  # bblup_auc <- auc(roc(true, psi_bblup[, 1]))

  # Bayesian lasso
  start_time <- Sys.time()
  ETA <- list(list(X = x_train, model = "BL"))
  reg.BL <- BGLR(y = y_train, ETA = ETA, nIter = mcmc.iter,
                 burnIn = mcmc.burn, verbose = FALSE)
  end_time <- Sys.time()
  bl_time <- difftime(end_time, start_time, units = "secs")[[1]]
  BL_beta <- reg.BL$ETA[[1]]$b
  pred <- as.matrix(x_test) %*% BL_beta
  diff_pred <- y_test - pred
  diff_trunc <- diff_pred[abs(diff_pred) < 3]
  bl_mse <- mean(diff_trunc ^ 2)
  # bl_mse <- mean((pred - f_test)^2) / mean((f_test - mean(f_test))^2)
  psi_bl <- BL_beta^2
  psi_bl <- psi_bl / max(psi_bl)
  bl_auc <- auc(roc(true, psi_bl))
  
  cat(c(as.character(name), n, dim, i, knock_auc, bnn_auc, bkmr_auc, bart_auc, gam_auc, earth_auc,
        bakr_auc, brr_auc, bl_auc), file = paste0("sim_auc_", data_name, ".txt"), append = T, "\n")
  cat(c(as.character(name), n, dim, i, knock_time, bnn_time, bkmr_time, bart_time, gam_time, earth_time, 
        bakr_time, brr_time, bl_time), file = paste0("sim_time_", data_name, ".txt"), append = T, "\n")
  cat(c(as.character(name), n, dim, i, bkmr_mse, earth_mse, bart_mse, gam_mse, bakr_mse, brr_mse, bl_mse),
      file = paste0("sim_mse_", data_name, ".txt"), append = T, "\n")
  
  return(filename)
}

