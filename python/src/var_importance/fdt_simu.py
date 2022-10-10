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
# from jax.interpreters import xla
import numpy as np
import csv
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import OneHotEncoder

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
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

    def calculate_f1(true, pred):
        f1_lst = []
        for l in range(pred.__len__()):
            f1_lst.append(f1_score(true, pred >= pred[l]))
        f1_lst.append(f1_score(true, pred >= (np.max(pred) + 1)))
        return np.max(f1_lst)

    # @title prepare_training_data
    def prepare_training_data(data, n_obs):
        df_train = data.head(n_obs)
        df_test = data.tail(40)

        x_train, y_train, f_train = df_train, df_train.pop("y"), df_train.pop("f")
        x_test, y_test, f_test = df_test, df_test.pop("y"), df_test.pop("f")

        x_train = x_train.to_numpy()
        x_test = x_test.to_numpy()

        y_train = y_train.to_numpy().reshape(-1, 1).ravel()
        y_test = y_test.to_numpy().reshape(-1, 1).ravel()
        return x_train, y_train, x_test, y_test

    def estimate_psi_NN(X, W1, W2, b1, sig2=0.01, n_samp=100, batch_size=100):
        """Calculate psi.

        Args:
            X: (np array) n x d input matrix.
            W1: (np array) d x K matrix indicating the weight matrix of the first layer.
            W2: (np array) K x 1 matrix indicating the weight matrix of the second layer.
            b1: (np array) vector of length K indicating the intercepts of the  first layer.

        Returns:
            psi: (np array) vector of length d indicating the estimated variable importance.
        """
        if X.ndim == 1:
            X = X[None, :]
        layer1 = X @ W1 + b1
        phi = layer1 * (layer1 > 0)
        phi_batches = split_into_batches(phi, batch_size)
        num_batch = phi_batches.__len__()
        weight_cov_val = np.float64(compute_inverse(phi_batches[0], sig_sq=sig2))

        for batch_id in range(1, num_batch):
            H_inv = weight_cov_val
            phi_batch = phi_batches[batch_id]
            weight_cov_val = minibatch_woodbury_update(phi_batch, H_inv)
        Sigma_beta = weight_cov_val * sig2

        try:
            beta_samp = np.random.multivariate_normal(W2[:, 0], Sigma_beta, size=n_samp)
        except:
            beta_samp = np.random.multivariate_normal(W2[:, 0], np.diag(np.diag(Sigma_beta)),
                                                      size=n_samp)  # (n_samp, D)

        drelu = (X @ W1 + b1 > 0).astype(float)
        layer2_samp = np.multiply(drelu[:, :, None], beta_samp.T)
        layer2 = np.mean(layer2_samp, axis=2)
        gradient = layer2 @ W1.T
        psi = np.mean(gradient ** 2, 0)
        return psi[None, :]

    # @title MetricsCallback
    class MetricsCallback(tf.keras.callbacks.Callback):
        def __init__(self, x_train, x_test, y_test, outlier_mse_cutoff=[10, 50, 100], sig2=0.01):
            super().__init__()

            self.history = {'psi_auroc': [], 'psi_auprc': []}
            self.history.update({f'y_mse_{thresh}': [] for thresh in outlier_mse_cutoff})
            self.x_train = x_train
            self.x_test = x_test
            self.y_test = y_test
            self.outlier_mse_cutoff = outlier_mse_cutoff
            self.sig2 = sig2
            self.psi_true = np.concatenate((np.repeat(1, 5), np.repeat(0, args.dim_in - 5)))

        def compute_psi_auc(self, n_samp=100, batch_size=100):
            W1 = self.model.layers[1].weights[0].numpy()  # d X K
            b1 = self.model.layers[1].bias.numpy()  # vector of length K
            W2 = self.model.layers[2].weights[0].numpy()  # K X 1
            psi_est = estimate_psi_NN(self.x_train, W1, W2, b1, sig2=self.sig2,
                                      n_samp=n_samp, batch_size=batch_size).flatten()
            auroc = roc_auc_score(self.psi_true, psi_est)
            auprc = average_precision_score(self.psi_true, psi_est)
            return auroc, auprc

        def compute_y_mse(self, outlier_mse_cutoff):
            y_est = self.model.predict(self.x_test, verbose=0).flatten()
            mse = (self.y_test - y_est) ** 2
            # Remove outlier MSEs.
            mse = mse[mse < outlier_mse_cutoff]
            return np.mean(mse)

        def on_epoch_end(self, epoch, logs=None):
            auroc, auprc = self.compute_psi_auc()

            for thresh in self.outlier_mse_cutoff:
                self.history[f'y_mse_{thresh}'].append(self.compute_y_mse(thresh))

            self.history['psi_auroc'].append(auroc)
            self.history['psi_auprc'].append(auprc)

    # @title OutlierRobustMSE
    class OutlierRobustMSE(tf.keras.metrics.Metric):
        def __init__(self, outlier_mse_cutoff=10., name='robust_mse', **kwargs):
            name = name + '_' + str(outlier_mse_cutoff)
            super().__init__(name=name, **kwargs)
            self.outlier_mse_cutoff = outlier_mse_cutoff
            self.mse = tf.keras.metrics.Mean()

        def update_state(self, y_true, y_pred, sample_weight):
            batch_mse = tf.math.square(y_true - y_pred)
            not_outlier = tf.less(batch_mse, self.outlier_mse_cutoff)

            batch_mse = tf.boolean_mask(batch_mse, not_outlier)

            self.mse.update_state(values=batch_mse, sample_weight=sample_weight)

        def result(self):
            return self.mse.result()

        def reset_state(self):
            self.mse.reset_state()

    # @title get_compiled_model
    def get_compiled_model(dim_in, dim_hidden=1024, seed=0, lr=1e-3,
                           l1=1e-2, l2=1e-2, outlier_mse_cutoff=[10.]):
        inputs = keras.Input(shape=(dim_in,), name="input")
        x = layers.Dense(dim_hidden,
                         kernel_initializer=keras.initializers.RandomNormal(seed=seed),
                         bias_initializer=keras.initializers.RandomUniform(seed=seed),
                         kernel_regularizer=keras.regularizers.L1L2(l1=l1, l2=l2),
                         bias_regularizer=keras.regularizers.L1L2(l1=l1, l2=l2),
                         activation="relu",
                         name="hidden")(inputs)
        outputs = layers.Dense(1, name="output")(x)

        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss=keras.losses.MeanSquaredError(),
            metrics=[
                OutlierRobustMSE(outlier_mse_cutoff=cutoff)
                for cutoff in outlier_mse_cutoff],
        )
        return model

    def split_into_batches(X, batch_size):
        return [X[i:i + batch_size] for i in range(0, len(X), batch_size)]

    def split_into_betas(betas_set, batch_size):
        return [betas_set[:, i:i + batch_size, :] for i in range(0, betas_set.shape[1], batch_size)]

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

        def estimate_psi(self, X, compute_cov=True, n_samp=100):
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


    ## allocate space for results
    res = {}

    # --------- Load data -----------
    data = pd.read_csv(os.path.join("featurized-decision-tree/python/experiments/expr/datasets", args.path,
                                    "{data}_n{n}_d{d}_i{i}.csv".format(data=args.dataset,
                                                                       n=args.n_obs, d=args.dim_in,
                                                                       i=args.rep)), index_col=0)
    x_train, y_train, x_test, y_test = prepare_training_data(data, args.n_obs)


    # --------- Train model -----------
    true = np.concatenate((np.repeat(1, 5), np.repeat(0, args.dim_in - 5)))
    n_samp = 100
    max_leaf_nodes = 2**(int(np.log2(np.sqrt(args.n_obs) * np.log(args.n_obs))) + 1)

    c_lst = [0.1, 1.0]
    for c in c_lst:
        extra = ExtraTreesRegressor(n_estimators=50, max_leaf_nodes=max_leaf_nodes, random_state=0).fit(x_train,
                                                                                                        y_train)
        rf_raw = RandomForestRegressor(n_estimators=50, max_leaf_nodes=max_leaf_nodes, random_state=0).fit(x_train,
                                                                                                           y_train)
        beta_set = np.zeros((extra.n_estimators, max_leaf_nodes))
        betas_set = np.zeros((extra.n_estimators, n_samp, max_leaf_nodes))
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
            try:
                beta_samp = np.random.multivariate_normal(fdt.beta, fdt.Sigma_beta, size=n_samp)
            except:
                beta_samp = np.random.multivariate_normal(fdt.beta,
                                                          np.diag(np.diag(fdt.Sigma_beta)),
                                                          size=n_samp)  # (n_samp, D)
            betas_set[j, :, :] = beta_samp
        # res['runtime_train'] = time.time() - start_time

        beta_set = jnp.asarray(beta_set)
        betas_set = jnp.asarray(betas_set)
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

        grad_fs = jax.jit(jax.vmap(jax.vmap(jax.vmap(jax.grad(predict, argnums=0),
                                                     in_axes=(None, 0, 0, 0, 0), out_axes=0),
                                            in_axes=(0, None, None, None, None), out_axes=0),
                                   in_axes=(None, None, None, None, 1), out_axes=0))

        batch_beta = 20
        batch_samp = 20
        batch_tree = 20
        beta_batches = split_into_betas(betas_set, batch_size=batch_beta)
        X_batches = split_into_batches(x_train, batch_size=batch_samp)
        map_matrix_batches = split_into_batches(map_matrix_set, batch_size=batch_tree)
        feature_batches = split_into_batches(feature_set, batch_size=batch_tree)
        threshold_batches = split_into_batches(threshold_set, batch_size=batch_tree)

        psi_beta = []
        for beta_id in range(beta_batches.__len__()):
            psi_samp = []
            beta_tree_batches = split_into_batches(beta_batches[beta_id], batch_size=batch_tree)
            for batch_id in range(X_batches.__len__()):
                psi_tree = []
                for tree_id in range(feature_batches.__len__()):
                    psi_tmp = np.array(
                        grad_fs(X_batches[batch_id], map_matrix_batches[tree_id], feature_batches[tree_id],
                                threshold_batches[tree_id], beta_tree_batches[tree_id]))
                    psi_tree.append(psi_tmp ** 2)
                psi_samp.append(np.median(np.concatenate(psi_tree, axis=2), axis=2))
            psi_beta.append(np.mean(np.concatenate(psi_samp, axis=1), axis=1))
        psi_est = np.mean(np.concatenate(psi_beta, axis=0), axis=0)

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
        res["f1_fdt_c{c}".format(c=c)] = calculate_f1(true, psi_est)
        res["f1_fdt_cat_c{c}".format(c=c)] = calculate_f1(true, psi_est_cat)
        res["roc_fdt_c{c}".format(c=c)] = roc_fdt
        res["roc_fdt_cat_c{c}".format(c=c)] = roc_fdt_cat



    # --------- Store results -----------
    res["f1_extra"] = calculate_f1(true, extra.feature_importances_)
    res["f1_rf_raw"] = calculate_f1(true, rf_raw.feature_importances_)
    res['roc_extra'] = roc_extra
    res['roc_rf_raw'] = roc_rf_raw

    # RFF
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
    res["f1_rfnn"] = calculate_f1(true, est_ind)
    res['roc_rfnn'] = roc_auc_score(true, est_ind)

    pred = m.predict(x_test)[0]
    tst_mse_rfnn = np.mean((y_test - pred) ** 2)

    # NN
    # Execute model run
    lr = 1e-3  # @param
    l1 = 1e2  # @param
    l2 = 1e2
    dim_hidden = 512  # @param
    outlier_mse_cutoff = [10, 50, 100]

    for i in range(10):
        model_nn = get_compiled_model(dim_in=args.dim_in, dim_hidden=dim_hidden,
                                      seed=i, lr=lr, l1=l1, l2=l2,
                                      outlier_mse_cutoff=outlier_mse_cutoff)

        # Define callbacks.
        early_stop_callback = keras.callbacks.EarlyStopping(
            monitor="val_robust_mse_50", min_delta=1e-6, patience=25, verbose=0)
        metrics_callback = MetricsCallback(x_train=x_train,
                                           x_test=x_test,
                                           y_test=y_test,
                                           outlier_mse_cutoff=outlier_mse_cutoff)

        model_nn.fit(x_train, y_train,
                     batch_size=64, epochs=500,
                     validation_data=(x_test, y_test),
                     callbacks=[early_stop_callback, metrics_callback],
                     verbose=0)

        # Decide best epoch by RMSE / AUC.
        mse_10_history = metrics_callback.history['y_mse_10']
        mse_50_history = metrics_callback.history['y_mse_50']
        mse_100_history = metrics_callback.history['y_mse_100']

        auroc_history = metrics_callback.history['psi_auroc']

        best_epoch_auroc = np.argmax(auroc_history)

        best_mse_10_by_auroc = mse_10_history[best_epoch_auroc]
        best_mse_50_by_auroc = mse_50_history[best_epoch_auroc]
        best_mse_100_by_auroc = mse_100_history[best_epoch_auroc]
        best_auroc_by_auroc = auroc_history[best_epoch_auroc]

        var_auroc_list.append(best_auroc_by_auroc)
        pred_mse_10_list.append(best_mse_10_by_auroc)
        pred_mse_50_list.append(best_mse_50_by_auroc)
        pred_mse_100_list.append(best_mse_100_by_auroc)

    res["f1_nn"] = var_f1_lst[np.argsort(pred_mse_lst)[len(pred_mse_lst) // 2]]
    res['roc_nn'] = var_auc_lst[np.argsort(pred_mse_lst)[len(pred_mse_lst) // 2]]
    res["f1_nn_best"] = var_f1_lst[np.argmin(pred_mse_lst)]
    res['roc_nn_best'] = var_auc_lst[np.argmin(pred_mse_lst)]

    res['lengthscale'] = l
    res['tst_mse_fdt'] = tst_mse_fdt
    res['tst_mse_extra'] = tst_mse_extra
    res['tst_mse_raw'] = tst_mse_raw
    res['tst_mse_rfnn'] = tst_mse_rfnn
    res['tst_mse_nn'] = np.median(pred_mse_lst)
    res['tst_mse_nn_best'] = np.min(pred_mse_lst)
    return res

if __name__ == '__main__':
    main()



