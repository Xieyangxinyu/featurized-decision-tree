# standard library imports
import os
import sys
import argparse
import time

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
import jax
import jax.numpy as jnp
from jax import jit
import tensorflow as tf
import numpy as np
import csv
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import kernelized as kernel_layers

def get_parser():
    parser = argparse.ArgumentParser()

    # general experiment arguments
    parser.add_argument('--dir_out', type=str, default='output/')
    parser.add_argument('--rep', type=int, default=1)

    # dataset arguments
    parser.add_argument('--dataset', type=str, default='rbf')
    parser.add_argument('--n_obs', type=int, default=10)
    parser.add_argument('--dim_in', type=int, default=2)
    parser.add_argument('--path', type=str, default='cat')
    parser.add_argument('--seed', type=int, default=1, help='seed for dataset')

    # general model argument
    parser.add_argument('--sig2', type=float, default=.01, help='observational noise')
    parser.add_argument('--c', type=float, default=10, help='smoothing parameter of soft tree')
    parser.add_argument('--lengthscale', type=float, default=5.0, help='lengthscale for random fourier features')

    return parser


def main(args=None):
    parser = get_parser()
    args = parser.parse_args(args)

    if not os.path.exists(args.dir_out):
        os.makedirs(args.dir_out)

    def split_into_batches(X, batch_size):
        return [X[i:i + batch_size] for i in range(0, len(X), batch_size)]

    def compute_inverse(X, sig_sq=1.):
        return np.linalg.inv(np.matmul(X.T, X) + sig_sq * np.identity(X.shape[1]))

    def minibatch_woodbury_update(X, H_inv):
        batch_size = X.shape[0]

        M0 = np.eye(batch_size, dtype=np.float64) + np.matmul(X, np.matmul(H_inv, X.T))
        M = np.linalg.inv(M0)
        B = np.matmul(X, H_inv)
        H_new = H_inv - np.matmul(B.T, np.matmul(M, B))
        return H_new

    def minibatch_interaction_update(Phi_y, rfnn_output, Y_batch):
        return Phi_y + np.matmul(rfnn_output.T, Y_batch)

    class FDT(object):
        def __init__(self, model, x_train, y_train, c=1.0, sig2=0.01):
            super().__init__()

            self.x_train = x_train
            self.y_train = y_train
            self.c = c
            self.sig2 = sig2
            self.feature = model.tree_.feature
            self.threshold = model.tree_.threshold
            self.children_left = model.tree_.children_left
            self.children_right = model.tree_.children_right
            self.stable_sigmoid = lambda x: jnp.exp(jax.nn.log_sigmoid(x))
            try:
                self.hidden_features = self.feature[jnp.where(self.children_left > 0)[0]]
                self.hidden_threshold = self.threshold[jnp.where(self.children_left > 0)[0]]
            except:
                self.hidden_features = self.feature[np.where(self.children_left > 0)[0]]
                self.hidden_threshold = self.threshold[np.where(self.children_left > 0)[0]]
            self.build_map_matrix()
            leaf_id = model.apply(x_train)
            enc = OneHotEncoder(handle_unknown="ignore")
            enc.fit(leaf_id.reshape(-1, 1))
            self.feature0 = enc.transform(leaf_id.reshape(-1, 1)).toarray()

        def train(self):
            feature_soft = self.compute_feature(self.x_train)
            feature_soft = feature_soft * self.feature0
            Sigma_tmp = jnp.diag(1 / (jnp.diag(jnp.matmul(feature_soft.T, feature_soft)) + self.sig2))
            feature_soft = jnp.matmul(feature_soft, Sigma_tmp)
            self.beta = jnp.matmul(feature_soft.T, self.y_train)
            self.Sigma_beta = self.sig2 * Sigma_tmp

        def compute_feature(self, x):
            hidden_features_matrix = x[:, self.hidden_features]
            right_indicator = self.stable_sigmoid(self.c * (hidden_features_matrix - self.hidden_threshold))
            left_indicator = 1. - right_indicator
            soft_indicator = jnp.concatenate([left_indicator, right_indicator], axis=1)
            F_leaf = jnp.multiply(soft_indicator[:, :, np.newaxis], self.map_matrix) + (1.0 - self.map_matrix)
            return jnp.prod(F_leaf, axis=1)

        def build_map_matrix(self):
            # build two dfs, one for storing mapping info for left children, one for right
            column_name = [str(id) for id in jnp.where(self.children_left < 0)[0]]
            row_name = [str(id) for id in jnp.where(self.children_left > 0)[0]]
            df_left = pd.DataFrame(np.zeros((len(row_name), len(column_name))),
                                   index=row_name, columns=column_name)
            df_right = pd.DataFrame(np.zeros((len(row_name), len(column_name))),
                                    index=row_name, columns=column_name)
            df_list = [df_left, df_right]

            # once reach leaf node, update dfs
            def process_df(path):
                leaf_node = str(path[-1][0])
                flag = path[-1][1]
                for element in path[:-1][::-1]:
                    df_list[flag][leaf_node].loc[str(element[0])] = 1.0
                    flag = element[1]

            # recursively obtain root to leaf path with direction
            def pre_order(index, flag, path, children_left, children_right):
                path.append((index, flag))
                if children_left[index] < 0:
                    process_df(path)
                if children_left[index] > 0:
                    pre_order(children_left[index], 0, path, children_left, children_right)
                if children_right[index] > 0:
                    pre_order(children_right[index], 1, path, children_left, children_right)
                del path[-1]

            flag = -1
            path = []
            pre_order(0, flag, path, self.children_left, self.children_right)
            self.map_matrix = jnp.concatenate([df_list[0].to_numpy(), df_list[1].to_numpy()], axis=0)

    class RFNN(object):
        def __init__(self, X, Y, dim_hidden=1024, sig2=0.01, lengthscale=1., seed=None):
            super().__init__()

            self.X = X
            self.Y = Y
            self.dim_in = X.shape[1]
            self.sig2 = sig2
            self.lengthscale = lengthscale
            self.dim_hidden = dim_hidden
            self.seed = seed

            self.rfnn_layer = kernel_layers.RandomFourierFeatures(
                output_dim=self.dim_hidden,
                kernel_initializer='gaussian',
                scale=self.lengthscale,
                seed=self.seed)
            self.rfnn_layer.build(input_shape=(None, self.dim_in))
            self.RFNN_weight = self.rfnn_layer.kernel
            self.RFNN_bias = self.rfnn_layer.bias

        def train(self, batch_size=20, epochs=1):
            ### Training and Evaluation ###
            X_batches = split_into_batches(self.X, batch_size) * epochs
            Y_batches = split_into_batches(self.Y, batch_size) * epochs

            num_steps = X_batches.__len__()
            num_batch = int(num_steps / epochs)
            rfnn_1 = tf.cast(self.rfnn_layer(X_batches[0]) * np.sqrt(2. / self.dim_hidden),
                            dtype=tf.float64).numpy()
            weight_cov_val = np.float64(compute_inverse(rfnn_1, sig_sq=self.sig2))
            covl_xy_val = np.float64(np.matmul(rfnn_1.T, Y_batches[0]))

            for batch_id in range(1, num_batch):
                H_inv = weight_cov_val
                Phi_y = covl_xy_val
                X_batch = X_batches[batch_id]
                Y_batch = Y_batches[batch_id]

                ## update posterior mean/covariance
                try:
                    rfnn_batch = tf.cast(self.rfnn_layer(X_batch) * np.sqrt(2. / self.dim_hidden),
                                        dtype=tf.float64).numpy()
                    weight_cov_val = minibatch_woodbury_update(rfnn_batch, H_inv)
                    covl_xy_val = minibatch_interaction_update(Phi_y, rfnn_batch, Y_batch)
                except:
                    print("\n================================\n"
                          "Problem occurred at Step {}\n"
                          "================================".format(batch_id))

            self.beta = np.matmul(weight_cov_val, covl_xy_val)
            self.weight_cov_val = weight_cov_val
            self.covl_xy_val = covl_xy_val
            self.Sigma_beta = weight_cov_val * self.sig2

        def make_rfnn_feature(self, X):
            return np.sqrt(2. / self.dim_hidden) * np.cos(
                np.matmul(X, self.RFNN_weight) + self.RFNN_bias)

        def predict(self, X):
            D = self.dim_hidden
            rfnn_new = np.sqrt(2. / D) * np.cos(np.matmul(X, self.RFNN_weight) +
                                               self.RFNN_bias)
            pred_mean = np.matmul(rfnn_new, self.beta)
            pred_cov = np.matmul(np.matmul(rfnn_new, self.Sigma_beta), rfnn_new.T)

            return pred_mean.reshape((-1, 1)), pred_cov

        def estimate_psi(self, X, compute_cov=False, n_samp=1000):
            nD_mat = np.sin(np.matmul(X, self.RFNN_weight) + self.RFNN_bias)
            n, d = X.shape
            D = self.RFNN_weight.shape[1]

            psi_mean = np.zeros(self.dim_in)
            psi_var = np.zeros(self.dim_in)

            if compute_cov:
                der_array = np.zeros((n, d, n_samp))
                try:
                    beta_samp = np.random.multivariate_normal(self.beta, self.Sigma_beta, size=n_samp).T
                except:
                    beta_samp = np.random.multivariate_normal(self.beta,
                                                              np.diag(np.diag(self.Sigma_beta)),
                                                              size=n_samp).T
                # (D, n_samp)
                for r in range(n):
                    cur_mat = np.diag(nD_mat[r, :])
                    cur_mat_W = np.matmul(self.RFNN_weight, cur_mat)  # (d, D)
                    cur_W_beta = np.matmul(cur_mat_W, beta_samp)  # (d, n_samp)
                    der_array[r, :, :] = cur_W_beta

                der_array = der_array * np.sqrt(2. / D)
                for l in range(self.dim_in):
                    grad_samp = der_array[:, l, :].T  # (n_samp, n)
                    psi_samp = np.mean(grad_samp ** 2, 1)
                    psi_mean[l] = np.mean(psi_samp)
                    psi_var[l] = np.var(psi_samp)
            else:
                mat_1 = np.diag(nD_mat[0, :])
                psi_mat = np.matmul(np.matmul(self.RFNN_weight, mat_1), self.beta).reshape(1,-1)
                for r in range(1, n):
                    cur_mat = np.diag(nD_mat[r, :])
                    cur_mat_W = np.matmul(self.RFNN_weight, cur_mat)
                    cur_W_beta = np.matmul(cur_mat_W, self.beta).reshape(1,-1)
                    psi_mat = np.concatenate((psi_mat, cur_W_beta))
                psi_mat = psi_mat * np.sqrt(2. / D)
                psi_mean = np.mean(psi_mat ** 2, 0)
            return psi_mean, psi_var


    # def plot_slice(x, y, quantile=.5, dim=0, ax=None, fix_x=0):
    #     '''
    #
    #     x: (N,D) training inputs
    #     y: (N,1) or (N,) training outputs
    #     quantile: Quantile of fixed x variables to use in plot
    #     dim: dimension of x to plot on x-axis
    #
    #     Everything should be numpy
    #     '''
    #
    #     if ax is None:
    #         fig, ax = plt.subplots()
    #
    #     # x-axis
    #     midx = (x[:, dim].min() + x[:, dim].max()) / 2
    #     dx = x[:, dim].max() - x[:, dim].min()
    #     x_plot = np.linspace(midx - .75 * dx, midx + .75 * dx, x.shape[0])
    #
    #     x_plot_all = np.quantile(x, q=quantile, axis=0)*np.ones((x_plot.shape[0], x.shape[1])) # use quantile
    #     # x_plot_all = np.zeros((x_plot.shape[0], x.shape[1]))  # use zeros
    #     x_plot_all[:, dim] = x_plot
    #
    #     # plot
    #     if fix_x == 0:
    #         color = "blue"
    #         label = "X1=0, X2=0"
    #         x_plot_all[:, 0] = 0
    #         x_plot_all[:, 1] = 0
    #     elif fix_x == 1:
    #         color = "green"
    #         label = "X1=1, X2=0"
    #         x_plot_all[:, 0] = 1
    #         x_plot_all[:, 1] = 0
    #     elif fix_x == 2:
    #         color = "purple"
    #         label = "X1=0, X2=1"
    #         x_plot_all[:, 0] = 0
    #         x_plot_all[:, 1] = 1
    #     else:
    #         color = "red"
    #         label = "X1=1, X2=1"
    #         x_plot_all[:, 0] = 1
    #         x_plot_all[:, 1] = 1
    #
    #     # sample from model
    #     f_samp_plot = np.array(f(x_plot_all, map_matrix_set, feature_set, threshold_set, beta_set)).T
    #
    #     ax.scatter(x[:, dim], y, color=color)  # training data
    #     ax.plot(x_plot, np.mean(f_samp_plot, 0), color=color, label=label)  # posterior mean
    #     for q in [.025, .05, .1]:
    #         ci = np.quantile(f_samp_plot, [q, 1 - q], axis=0)
    #         ax.fill_between(x_plot_all[:, dim].reshape(-1), ci[0, :], ci[1, :], alpha=.1, color=color)
    #
    # def plot_slices_5(x_all, y_all, quantile=.5, figsize=(4, 4)):
    #     dim_in = 3
    #     fig, ax = plt.subplots(4, dim_in, figsize=figsize, sharex=True, sharey=True)
    #
    #     # fig.suptitle("1d slices")
    #     for fix_x in range(4):
    #         for dim in range(2, 5):
    #             ax_dim = ax[fix_x, dim - 2] if dim_in > 1 else ax
    #             plot_slice(x_all[fix_x], y_all[fix_x].ravel(), quantile=quantile, dim=dim, ax=ax_dim, fix_x=fix_x)
    #             if fix_x == 3:
    #                 ax_dim.set_xlabel("x" + str(dim + 1))
    #         ax[fix_x, 2].legend(loc="upper right")
    #
    #     fig.text(0.06, 0.5, "y", va="center", rotation="vertical")
    #
    #     return fig, ax



    ## allocate space for results
    res = {}

    # --------- Load data -----------
    data = pd.read_csv(os.path.join("/n/home10/irisdeng/FDT/python/experiments/expr/datasets", args.path,
                                    "{data}_n{n}_d{d}_i{i}.csv".format(data=args.dataset,
                                                                       n=args.n_obs, d=args.dim_in,
                                                                       i=args.rep)), index_col=0)
    df_train = data.head(args.n_obs)
    df_test = data.tail(40)

    x_train, y_train, f_train = df_train, df_train.pop("y"), df_train.pop("f")
    x_test, y_test, f_test = df_test, df_test.pop("y"), df_test.pop("f")

    x_train = x_train.to_numpy()
    x_test = x_test.to_numpy()

    y_train = y_train.to_numpy().reshape(-1, 1).ravel()
    y_test = y_test.to_numpy().reshape(-1, 1).ravel()

    # --------- Train model -----------
    true = np.concatenate((np.repeat(1, 5), np.repeat(0, args.dim_in - 5)))
    max_leaf_nodes = 2**(int(np.log2(np.sqrt(args.n_obs) * np.log(args.n_obs))) + 1)

    c_lst = [0.1, 1.0]
    for c in c_lst:
        extra = ExtraTreesRegressor(n_estimators=50, max_leaf_nodes=max_leaf_nodes, random_state=0).fit(x_train,
                                                                                                        y_train)
        rf_raw = RandomForestRegressor(n_estimators=50, max_leaf_nodes=max_leaf_nodes, random_state=0).fit(x_train,
                                                                                                           y_train)
        beta_set = np.zeros((extra.n_estimators, max_leaf_nodes))
        map_matrix_set = np.zeros((extra.n_estimators, 2 * (max_leaf_nodes - 1), max_leaf_nodes))
        feature_set = np.zeros((extra.n_estimators, max_leaf_nodes - 1))
        threshold_set = np.zeros((extra.n_estimators, max_leaf_nodes - 1))

        # start_time = time.time()
        for j, model in enumerate(extra.estimators_):
            fdt = FDT(model, x_train, y_train, c=c)
            fdt.train()
            beta_set[j, :] = fdt.beta
            map_matrix_set[j, :, :] = fdt.map_matrix
            feature_set[j, :] = fdt.hidden_features
            threshold_set[j, :] = fdt.hidden_threshold
        # res['runtime_train'] = time.time() - start_time

        beta_set = jnp.asarray(beta_set)
        map_matrix_set = jnp.asarray(map_matrix_set)
        feature_set = jnp.asarray(feature_set, dtype=int)
        threshold_set = jnp.asarray(threshold_set)

        stable_sigmoid = lambda x: jnp.exp(jax.nn.log_sigmoid(x))

        def predict(x, map_matrix, feature, threshold, beta):
            hidden_features = jnp.asarray(x)[feature]
            right_indicator = stable_sigmoid(c * (hidden_features - threshold))
            left_indicator = 1. - right_indicator
            soft_indicator = jnp.concatenate([left_indicator, right_indicator], axis=0)
            F_leaf = jnp.multiply(soft_indicator[:, np.newaxis], map_matrix) + (1.0 - map_matrix)
            soft_feature = jnp.prod(F_leaf, axis=0)
            return jnp.dot(soft_feature, beta)

        f = jax.jit(jax.vmap(jax.vmap(predict, in_axes=(None, 0, 0, 0, 0), out_axes=0),
                             in_axes=(0, None, None, None, None), out_axes=0))

        grad_f = jax.jit(jax.vmap(jax.vmap(jax.grad(predict, argnums=0),
                                           in_axes=(None, 0, 0, 0, 0), out_axes=0),
                                  in_axes=(0, None, None, None, None), out_axes=0))

        batch_size = 100
        if x_train.shape[0] > batch_size:
            X_batches = split_into_batches(x_train, batch_size=batch_size)
            psi_est_all = np.array(grad_f(X_batches[0], map_matrix_set, feature_set, threshold_set, beta_set))
            for batch_id in range(1, X_batches.__len__()):
                X_batch = X_batches[batch_id]
                psi_est_tmp = np.array(grad_f(X_batch, map_matrix_set, feature_set, threshold_set, beta_set))
                psi_est_all = np.concatenate([psi_est_all, psi_est_tmp], axis=0)
        else:
            psi_est_all = np.array(grad_f(x_train, map_matrix_set, feature_set, threshold_set, beta_set))

        grad_train = np.mean(psi_est_all ** 2, axis=0)
        psi_est = np.median(grad_train, axis=0)

        psi_est_cat = np.copy(psi_est)
        x1_1 = np.concatenate([np.ones(x_train.shape[0]).reshape((-1, 1)), x_train[:, 1:]], axis=1)
        x1_0 = np.concatenate([np.zeros(x_train.shape[0]).reshape((-1, 1)), x_train[:, 1:]], axis=1)
        pred_11 = np.array(f(x1_1, map_matrix_set, feature_set, threshold_set, beta_set))
        pred_10 = np.array(f(x1_0, map_matrix_set, feature_set, threshold_set, beta_set))
        contrast1 = np.mean((np.mean(pred_11, axis=1) - np.mean(pred_10, axis=1)) ** 2)
        x2_1 = np.concatenate(
            [x_train[:, 0].reshape((-1, 1)), np.ones(x_train.shape[0]).reshape((-1, 1)), x_train[:, 2:]],
            axis=1)
        x2_0 = np.concatenate(
            [x_train[:, 0].reshape((-1, 1)), np.zeros(x_train.shape[0]).reshape((-1, 1)), x_train[:, 2:]],
            axis=1)
        pred_21 = np.array(f(x2_1, map_matrix_set, feature_set, threshold_set, beta_set))
        pred_20 = np.array(f(x2_0, map_matrix_set, feature_set, threshold_set, beta_set))
        contrast2 = np.mean((np.mean(pred_21, axis=1) - np.mean(pred_20, axis=1)) ** 2)

        x6_1 = np.concatenate([x_train[:, :5], np.ones(x_train.shape[0]).reshape((-1, 1)), x_train[:, 6:]], axis=1)
        x6_0 = np.concatenate([x_train[:, :5], np.zeros(x_train.shape[0]).reshape((-1, 1)), x_train[:, 6:]], axis=1)
        pred_61 = np.array(f(x6_1, map_matrix_set, feature_set, threshold_set, beta_set))
        pred_60 = np.array(f(x6_0, map_matrix_set, feature_set, threshold_set, beta_set))
        contrast6 = np.mean((np.mean(pred_61, axis=1) - np.mean(pred_60, axis=1)) ** 2)

        x7_1 = np.concatenate([x_train[:, :6], np.ones(x_train.shape[0]).reshape((-1, 1)), x_train[:, 7:]],
                              axis=1)
        x7_0 = np.concatenate([x_train[:, :6], np.zeros(x_train.shape[0]).reshape((-1, 1)), x_train[:, 7:]],
                              axis=1)
        pred_71 = np.array(f(x7_1, map_matrix_set, feature_set, threshold_set, beta_set))
        pred_70 = np.array(f(x7_0, map_matrix_set, feature_set, threshold_set, beta_set))
        contrast7 = np.mean((np.mean(pred_71, axis=1) - np.mean(pred_70, axis=1)) ** 2)

        psi_est_cat[0] = contrast1
        psi_est_cat[1] = contrast2
        psi_est_cat[5] = contrast6
        psi_est_cat[6] = contrast7

        f_test_pred = np.array(f(x_test, map_matrix_set, feature_set, threshold_set, beta_set))
        tst_mse_fdt = np.mean((y_test - np.mean(f_test_pred, 1)) ** 2)
        pred = extra.predict(x_test)
        tst_mse_extra = np.mean((y_test - pred) ** 2)
        pred_raw = rf_raw.predict(x_test)
        tst_mse_raw = np.mean((y_test - pred_raw) ** 2)

        roc_fdt = roc_auc_score(true, psi_est)
        roc_fdt_cat = roc_auc_score(true, psi_est_cat)
        roc_extra = roc_auc_score(true, extra.feature_importances_)
        roc_rf_raw = roc_auc_score(true, rf_raw.feature_importances_)
        res["roc_fdt_c{c}".format(c=c)] = roc_fdt
        res["roc_fdt_cat_c{c}".format(c=c)] = roc_fdt_cat



    # --------- Store results -----------
    res['roc_extra'] = roc_extra
    res['roc_rf_raw'] = roc_rf_raw


    # categorical interacts with continuous
    # x_all =[]
    # y_all = []
    # x1 = np.copy(x_train)
    # x1 = x1[(x1[:, 0] == 0) & (x1[:, 1] == 0), :]
    # n_size = np.min([x1.shape[0], 50])
    # x1 = x1[np.random.choice(x1.shape[0], size=n_size, replace=False), :]
    # y1_all = np.array(f(x1, map_matrix_set, feature_set, threshold_set, beta_set))
    # y11 = np.mean(y1_all, axis=1).reshape(-1, 1)
    # x_all.append(x1)
    # y_all.append(y11)
    #
    # x1 = np.copy(x_train)
    # x1 = x1[(x1[:, 0] == 1) & (x1[:, 1] == 0), :]
    # n_size = np.min([x1.shape[0], 50])
    # x1 = x1[np.random.choice(x1.shape[0], size=n_size, replace=False), :]
    # y1_all = np.array(f(x1, map_matrix_set, feature_set, threshold_set, beta_set))
    # y10 = np.mean(y1_all, axis=1).reshape(-1, 1)
    # x_all.append(x1)
    # y_all.append(y10)
    #
    # x1 = np.copy(x_train)
    # x1 = x1[(x1[:, 0] == 0) & (x1[:, 1] == 1), :]
    # n_size = np.min([x1.shape[0], 50])
    # x1 = x1[np.random.choice(x1.shape[0], size=n_size, replace=False), :]
    # y1_all = np.array(f(x1, map_matrix_set, feature_set, threshold_set, beta_set))
    # y21 = np.mean(y1_all, axis=1).reshape(-1, 1)
    # x_all.append(x1)
    # y_all.append(y21)
    #
    # x1 = np.copy(x_train)
    # x1 = x1[(x1[:, 0] == 1) & (x1[:, 1] == 1), :]
    # n_size = np.min([x1.shape[0], 50])
    # x1 = x1[np.random.choice(x1.shape[0], size=n_size, replace=False), :]
    # y1_all = np.array(f(x1, map_matrix_set, feature_set, threshold_set, beta_set))
    # y20 = np.mean(y1_all, axis=1).reshape(-1, 1)
    # x_all.append(x1)
    # y_all.append(y20)
    #
    # fig, ax = plot_slices_5(x_all, y_all, quantile=.5, figsize=(10, 12))
    # fig.savefig(os.path.join("/n/home10/irisdeng/FDT/python/experiments/expr/results", args.path,
    #                          "slices_post/{name}_n{n}_d{dim_in}_rep{rep}_c{c}.png".format(name=args.dataset,
    #                                                                                       n=args.n_obs,
    #                                                                                       dim_in=args.dim_in,
    #                                                                                       rep=args.rep,
    #                                                                                       c=c)))
    #
    # plt.close('all')
    #
    # # uncertainty quantification of variable importance, continuous data
    # fig2, ax2 = plt.subplots()
    # ax2.set_title("{name}_n{n}_d{dim_in}".format(name=args.dataset, dim_in=args.dim_in, n=args.n_obs))
    # ax2.plot(np.arange(1, args.dim_in + 1), psi_est, color='blue')  # posterior mean
    # q = 0.05
    # ci = np.quantile(grad_train, [q, 1 - q], axis=0)
    # ax2.fill_between(np.arange(1, args.dim_in + 1), ci[0, :], ci[1, :], alpha=.1, color='blue')
    # fig2.savefig(os.path.join("/n/home10/irisdeng/FDT/python/experiments/expr/results", args.path,
    #                           "uq_vi/{name}_n{n}_d{dim_in}_rep{rep}_c{c}.png".format(name=args.dataset,
    #                                                                                  n=args.n_obs,
    #                                                                                  dim_in=args.dim_in,
    #                                                                                  rep=args.rep,
    #                                                                                  c=c)))
    # plt.close('all')

    # RFNN
    n_rfnn = int(np.sqrt(args.n_obs) * np.log(args.n_obs)) + 1
    pred_mse_rfnn = []
    l_lst = [5.0, 10.0, 16.0, 23.0]
    for ll in l_lst:
        m = RFNN(x_train, y_train, dim_hidden=n_rfnn, sig2=0.01, lengthscale=ll, seed=0)
        m.train()
        pred = m.predict(x_test)[0]
        pred_mse_rfnn.append(np.mean((pred - y_test) ** 2) / np.mean((y_test - np.mean(y_test)) ** 2))

    l = l_lst[np.argmin(pred_mse_rfnn)]
    m = RFNN(x_train, y_train, dim_hidden=n_rfnn, sig2=0.01, lengthscale=l, seed=0)
    m.train()
    psi_rfnn = m.estimate_psi(x_train)[0]
    est_ind = psi_rfnn / np.amax(psi_rfnn)
    res['roc_rfnn'] = roc_auc_score(true, est_ind)

    pred = m.predict(x_test)[0]
    tst_mse_rfnn = np.mean((y_test - pred) ** 2)
    res['lengthscale'] = l
    res['tst_mse_fdt'] = tst_mse_fdt
    res['tst_mse_extra'] = tst_mse_extra
    res['tst_mse_raw'] = tst_mse_raw
    res['tst_mse_rfnn'] = tst_mse_rfnn
    return res

if __name__ == '__main__':
    main()


# def estimate_psi(X, W1, W2, b1):
#     """Calculate psi.
#
#     Args:
#         X: (np array) n x d input matrix.
#         W1: (np array) d x K matrix indicating the weight matrix of the first layer.
#         W2: (np array) K x 1 matrix indicating the weight matrix of the second layer.
#         b1: (np array) vector of length K indicating the intercepts of the  first layer.
#
#     Returns:
#         psi: (np array) vector of length d indicating the estimated variable importance.
#     """
#     if X.ndim == 1:
#         X = X[None, :]
#     drelu = (X@W1 + b1 > 0).astype(float)
#     layer2 = np.multiply(drelu, W2[:,0])
#     gradient = layer2@W1.T
#     psi = np.mean(gradient ** 2, 0)
#     return psi

