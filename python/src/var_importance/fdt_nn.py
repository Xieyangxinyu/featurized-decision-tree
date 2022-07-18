import os
import sys
import argparse
import time
import pandas as pd
import numpy as np
import csv

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import roc_auc_score

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
    dim_hidden = 1024

    if not os.path.exists(args.dir_out):
        os.makedirs(args.dir_out)

    def get_compiled_model(seed=0, lr=1e-3, l1=1e-2):
        inputs = keras.Input(shape=(args.dim_in,), name="input")
        x = layers.Dense(dim_hidden, kernel_initializer=keras.initializers.RandomNormal(seed=seed),
                         bias_initializer=keras.initializers.RandomUniform(seed=seed),
                         kernel_regularizer=keras.regularizers.L1(l1=l1),
                         activation="relu", name="hidden")(inputs)
        outputs = layers.Dense(1, name="output")(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=keras.optimizers.SGD(learning_rate=lr),
            loss=keras.losses.MeanSquaredError(),
            metrics=keras.metrics.MeanSquaredError(),
        )
        return model

    def estimate_psi(X, W1, W2, b1):
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
        drelu = (X @ W1 + b1 > 0).astype(float)
        layer2 = np.multiply(drelu, W2[:, 0])
        gradient = layer2 @ W1.T
        psi = np.mean(gradient ** 2, 0)
        return psi[None, :]

    callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=1e-2, patience=2, verbose=1)]

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
    lr_lst = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    l1_lst = [1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
    tst_lst = []
    auc_lst = []
    for lr in lr_lst:
        for l1 in l1_lst:
            model = get_compiled_model(seed=0, lr=lr, l1=l1)
            model.fit(x_train, y_train, batch_size=32, epochs=500,
                      validation_data=(x_test, y_test), callbacks=callbacks)
            W1 = model.layers[1].weights[0].numpy()  # d X K
            b1 = model.layers[1].bias.numpy()  # vector of length K

            W2 = model.layers[2].weights[0].numpy()  # K X 1
            psi_est_all = estimate_psi(x_train, W1, W2, b1)
            pred_all = model.predict(x_test)
            for i in range(1, 10):
                model = get_compiled_model(seed=i, lr=lr, l1=l1)
                model.fit(x_train, y_train, batch_size=32, epochs=500,
                          validation_data=(x_test, y_test), callbacks=callbacks)
                W1 = model.layers[1].weights[0].numpy()  # d X K
                b1 = model.layers[1].bias.numpy()  # vector of length K

                W2 = model.layers[2].weights[0].numpy()  # K X 1
                psi_est_tmp = estimate_psi(x_train, W1, W2, b1)
                psi_est_all = np.concatenate([psi_est_all, psi_est_tmp], axis=0)
                pred_tmp = model.predict(x_test)
                pred_all = np.concatenate([pred_all, pred_tmp], axis=1)

            psi_est = np.median(psi_est_all, axis=0)
            pred_est = np.median(pred_all, axis=1)
            auc_lst.append(roc_auc_score(true, psi_est))
            tst_lst.append(np.mean((y_test - pred_est) ** 2))
    res['roc_nn'] = auc_lst[np.argmin(tst_lst)]
    res['tst_mse_nn'] = np.min(tst_lst)
    res['ind'] = np.argmin(tst_lst)
    return res

if __name__ == '__main__':
    main()


# plt.style.use('ggplot')
# plt.plot(hist.history['loss'], label = 'loss')
# plt.plot(hist.history['val_loss'], label='val loss')
# plt.title("Loss vs Val_Loss")
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.legend()
# plt.show()


