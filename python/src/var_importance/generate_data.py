# standard library imports
import os
import sys
import seaborn as sn
import matplotlib.pyplot as plt


# package imports
import autograd.numpy as np
from autograd import grad
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA


def distance_matrix(A, B, squared=False):
    """
    Compute all pairwise distances between vectors in A and B.

    Parameters
    ----------
    A : np.array
        shape should be (M, K)
    B : np.array
        shape should be (N, K)

    Returns
    -------
    D : np.array
        A matrix D of shape (M, N).  Each entry in D i,j represents the
        distance between row i in A and row j in B.
    """
    M = A.shape[0]
    N = B.shape[0]

    assert A.shape[1] == B.shape[1], f"The number of components for vectors in A \
        {A.shape[1]} does not match that of B {B.shape[1]}!"

    A_dots = (A * A).sum(axis=1).reshape((M, 1)) * np.ones(shape=(1, N))
    B_dots = (B * B).sum(axis=1) * np.ones(shape=(M, 1))
    D_squared = A_dots + B_dots - 2 * np.dot(A, B.T)

    if squared == False:
        D_squared = np.where(D_squared > 0.0, D_squared, 0.0)
        return np.sqrt(D_squared)

    return D_squared


class Dataset:
    x_train = None  # inputs
    z_train = None  # covariates
    f_train = None  # ground truth function
    y_train = None  # observed outcomes
    psi_train = None  # variable importance

    x_test = None
    z_test = None
    f_test = None
    y_test = None
    psi_test = None


class Toy(Dataset):
    def __init__(self,
                 f,
                 x_train,
                 x_test=None,
                 noise_sig2=1.0,
                 snr=2,
                 l=1,
                 seed=0,
                 standardize=True):

        self.f = lambda x: f(x).reshape(-1, 1)  # makes sure output is (n,1)
        self.noise_sig2 = noise_sig2
        self.snr = snr
        self.l = l
        self.dim_in = x_train.shape[1]

        # train
        self.x_train = x_train
        self.n_train = x_train.shape[0]

        # test
        self.x_test = x_test
        self.n_test = None if x_test is None else x_test.shape[0]

        # for categorical data
        self.x_train_cat = x_train[:, [0, 1, 5, 6]]
        self.x_test_cat = x_test[:, [0, 1, 5, 6]]

        self.evaluate_f()
        self.sample_y(seed)

        self.standardized = False
        if standardize:
            self.standardize()
        try:
            self.evaluate_psi()  # note: psi always based on whether data originally standardized
        except:
            print('Unable to compute variable importance, possible autograd not imported')

    def evaluate_f(self):

        scale_factor = self.snr * np.sqrt(self.noise_sig2)
        # train
        self.f_train = self.f(self.x_train).reshape(-1, 1)
        scaler = preprocessing.StandardScaler(with_mean=False).fit(self.f_train)
        f_train_scaled = scaler.transform(self.f_train)
        self.f_train = scale_factor * f_train_scaled

        # test
        if self.x_test is not None:
            self.f_test = self.f(self.x_test).reshape(-1, 1)
            scaler = preprocessing.StandardScaler(with_mean=False).fit(self.f_test)
            f_test_scaled = scaler.transform(self.f_test)
            self.f_test = scale_factor * f_test_scaled

    def sample_y(self, seed=0):
        r_noise = np.random.RandomState(seed)

        # train
        noise = r_noise.randn(self.n_train, 1) * np.sqrt(self.noise_sig2)
        self.y_train = self.f_train + noise

        # test
        if self.x_test is not None:
            noise_test = r_noise.randn(self.n_test, 1) * np.sqrt(self.noise_sig2)
            self.y_test = self.f_test + noise_test

    def evaluate_psi(self):
        grad_f = grad(self.f)
        jac_f_train = np.zeros((self.n_train, self.dim_in))
        for i in range(self.n_train):
            jac_f_train[i, :] = grad_f(self.x_train[i, :].reshape(1, -1))
        self.psi_train = np.mean(jac_f_train ** 2, 0)

        if self.x_test is not None:
            jac_f_test = np.zeros((self.n_test, self.dim_in))
            for i in range(self.n_test):
                jac_f_test[i, :] = grad_f(self.x_test[i, :].reshape(1, -1))
            self.psi_test = np.mean(jac_f_test ** 2, 0)

    def standardize(self):
        zscore = lambda x, mu, sigma: (x - mu.reshape(1, -1)) / sigma.reshape(1, -1)
        un_zscore = lambda x, mu, sigma: x * sigma.reshape(1, -1) + mu.reshape(1, -1)

        if not self.standardized:

            self.mu_x = np.mean(self.x_train, axis=0)
            self.sigma_x = np.std(self.x_train, axis=0)

            # self.mu_f = np.mean(self.f_train, axis=0)
            # self.sigma_f = np.std(self.f_train, axis=0)

            self.mu_y = np.mean(self.y_train, axis=0)
            self.sigma_y = np.std(self.y_train, axis=0)

            self.x_train = zscore(self.x_train, self.mu_x, self.sigma_x)
            if self.x_test is not None:
                self.x_test = zscore(self.x_test, self.mu_x, self.sigma_x)

            self.f_train = zscore(self.f_train, self.mu_y, self.sigma_y)
            if self.f_test is not None:
                self.f_test = zscore(self.f_test, self.mu_y, self.sigma_y)

            self.y_train = zscore(self.y_train, self.mu_y, self.sigma_y)
            if self.y_test is not None:
                self.y_test = zscore(self.y_test, self.mu_y, self.sigma_y)

            self.f_orig = self.f
            self.f = lambda x: zscore(self.f_orig(un_zscore(x, self.mu_x, self.sigma_x)), self.mu_y, self.sigma_y)
            self.standardized = True

    def unstandardize(self):
        zscore = lambda x, mu, sigma: (x - mu.reshape(1, -1)) / sigma.reshape(1, -1)
        un_zscore = lambda x, mu, sigma: x * sigma.reshape(1, -1) + mu.reshape(1, -1)

        if self.standardized:

            self.x_train = un_zscore(self.x_train, self.mu_x, self.sigma_x)
            if self.x_test is not None:
                self.x_test = un_zscore(self.x_test, self.mu_x, self.sigma_x)

            self.f_train = un_zscore(self.f_train, self.mu_y, self.sigma_y)
            if self.f_test is not None:
                self.f_test = un_zscore(self.f_test, self.mu_y, self.sigma_y)

            self.y_train = un_zscore(self.y_train, self.mu_y, self.sigma_y)
            if self.y_test is not None:
                self.y_test = un_zscore(self.y_test, self.mu_y, self.sigma_y)

            self.f = self.f_orig
            self.standardized = False


def gen_y(x_all, n_train, dim_in, seed_x, noise_sig2, version, snr, l, seed_noise):

    x_train = x_all[:n_train, :]
    x_test = x_all[n_train:, :]

    assert dim_in >= 5

    if version == 1:
        f = lambda x: x[:, 0] - x[:, 1] + x[:, 2] + 0.5 * x[:, 3] + 2 * x[:, 4]

    elif version == 2:
        r_f = np.random.RandomState(seed_x)
        # sample 5D function
        n_obs_sample = 500
        x0 = r_f.uniform(-2, 2, (n_obs_sample, 5))
        x0[:, [0, 1]] = r_f.binomial(1, 0.5, size=n_obs_sample * 2).reshape((n_obs_sample, 2))
        act = lambda z: np.exp(-distance_matrix(z, x0) ** 2 / (2 * l ** 2))
        rbf_kern = act(x0)
        f0 = r_f.multivariate_normal(np.zeros(n_obs_sample), rbf_kern, 1).reshape(-1, 1)
        y0 = f0 + r_f.normal(0, np.sqrt(noise_sig2), (n_obs_sample, 1))
        alpha = np.linalg.inv(rbf_kern + noise_sig2 * np.identity(x0.shape[0])) @ y0
        f = lambda z: act(z[:, :5]) @ alpha

    elif version == 3:
        r_f = np.random.RandomState(seed_x)
        # sample 5D function
        n_obs_sample = 500
        x0 = r_f.uniform(-2, 2, (n_obs_sample, 5))
        x0[:, [0, 1]] = r_f.binomial(1, 0.5, size=n_obs_sample * 2).reshape((n_obs_sample, 2))
        act = lambda z: (1 + np.sqrt(3) * distance_matrix(z, x0) / l) * np.exp(-np.sqrt(3) * distance_matrix(z, x0) / l)
        mat32_kern = act(x0)
        f0 = r_f.multivariate_normal(np.zeros(n_obs_sample), mat32_kern, 1).reshape(-1, 1)
        y0 = f0 + r_f.normal(0, np.sqrt(noise_sig2), (n_obs_sample, 1))
        alpha = np.linalg.inv(mat32_kern + noise_sig2 * np.identity(x0.shape[0])) @ y0
        f = lambda z: act(z[:, :5]) @ alpha

    elif version == 4:
        f = lambda x: np.sin(np.max([x[:, 0], x[:, 1]]) + np.arctan(x[:, 1])) / (1 + x[:, 0] + x[:, 4]) + \
                      np.sin(0.5 * x[:, 2]) * (1 + np.exp(x[:, 3] - 0.5 * x[:, 2])) + x[:, 2] ** 2 + \
                      2 * np.sin(x[:, 3]) + 4 * x[:, 4]
    
    return Toy(f, x_train, x_test, noise_sig2, snr, l, seed_noise)

def uncer_quan(dim_in, noise_sig2, snr, l, n_train, n_test=40, seed_x=0, seed_noise=0, version=1, base_data = None):
    # sample x
    r_x = np.random.RandomState(seed_x)
    n_all = n_train + n_test
    x_all = r_x.uniform(-2, 2, (n_all, dim_in))
    x_all[:, [0, 1, 5, 6]] = r_x.binomial(1, 0.5, size=n_all * 4).reshape((n_all, 4))

    return gen_y(x_all, n_train, dim_in, seed_x, noise_sig2, version, snr, l, seed_noise)


## continuous data
def uncer_quan_continuous(dim_in, noise_sig2, snr, l, n_train, n_test=40, seed_x=0, seed_noise=0, version=1, base_data = None):
    # sample x
    r_x = np.random.RandomState(seed_x)
    n_all = n_train + n_test
    x_all = r_x.uniform(-2, 2, (n_all, dim_in))

    return gen_y(x_all, n_train, dim_in, seed_x, noise_sig2, version, snr, l, seed_noise)


# simulation using features from real datasets
def simu_real(dim_in, noise_sig2, snr, l, n_train, n_test=40, seed_x=0, seed_noise=0, version=1, base_data = None):
    # sample x
    assert base_data is not None
    dim_original = base_data.shape[1]
    r_x = np.random.RandomState(seed_x)
    n_all = n_train + n_test
    x_original = base_data[np.random.choice(base_data.shape[0], size=n_all, replace=False), :]
    x_rest = r_x.uniform(-2, 2, (n_all, dim_in - dim_original))
    x_all = np.concatenate((x_original, x_rest), axis=1)

    return gen_y(x_all, n_train, dim_in, seed_x, noise_sig2, version, snr, l, seed_noise)


def load_dataset(name, type, dim_in, noise_sig2, snr, l, n_train, n_test=40, seed=0, base_data = None):
    '''
    inputs:

    returns:
    '''

    if name == 'linear':
        dataset = type(dim_in, noise_sig2, snr, l, n_train, n_test=n_test, seed_x=seed, seed_noise=seed,
                             version=1, base_data = base_data)

    elif name == 'rbf':
        dataset = type(dim_in, noise_sig2, snr, l, n_train, n_test=n_test, seed_x=seed, seed_noise=seed,
                             version=2, base_data = base_data)

    elif name == 'matern32':
        dataset = type(dim_in, noise_sig2, snr, l, n_train, n_test=n_test, seed_x=seed, seed_noise=seed,
                             version=3, base_data = base_data)

    elif name == 'complex':
        dataset = type(dim_in, noise_sig2, snr, l, n_train, n_test=n_test, seed_x=seed, seed_noise=seed,
                             version=4, base_data = base_data)

    return dataset


data_lst = ["linear", "rbf", "matern32", "complex"]
n_lst = [100, 200, 500, 1000]
dim_lst = [25, 50, 100, 200]

# generate synthetic-mixture datasets
for data in data_lst:
    for n in n_lst:
        for dim in dim_lst:
            for i in range(20):
                ds = load_dataset(data, uncer_quan, dim_in=dim, noise_sig2=.01, snr=2, l=1, n_train=n, seed=i)
                x_train = np.concatenate(
                    [ds.x_train_cat[:, [0, 1]], ds.x_train[:, [2, 3, 4]], ds.x_train_cat[:, [2, 3]], ds.x_train[:, 7:]],
                    axis=1)
                x_test = np.concatenate(
                    [ds.x_test_cat[:, [0, 1]], ds.x_test[:, [2, 3, 4]], ds.x_test_cat[:, [2, 3]], ds.x_test[:, 7:]],
                    axis=1)
                x = np.concatenate((x_train, x_test))
                # x = np.concatenate((ds.x_train, ds.x_test))
                y = np.concatenate((ds.y_train, ds.y_test))
                f = np.concatenate((ds.f_train, ds.f_test))
                df = np.concatenate((f, y, x), axis=1)

                columns = ["f", "y"]
                for j in range(dim):
                    columns.append("x" + str(j + 1))

                df_total = pd.DataFrame(df, columns=columns)
                df_total = df_total.astype(np.float32)
                psi = np.concatenate((ds.psi_train.reshape(1, -1), ds.psi_test.reshape(1, -1)))
                psi = pd.DataFrame(psi)
                df_total.to_csv(
                    "./python/experiments/expr/datasets/cat/{data}_n{n}_d{d}_i{i}.csv".format(data=data, n=n, d=dim, i=i))
                psi.to_csv(
                    "./python/experiments/expr/datasets/cat/psi/{data}_n{n}_d{d}_i{i}.csv".format(data=data, n=n, d=dim, i=i))

# generate synthetic-continuous datasets
for data in data_lst:
    for n in n_lst:
        for dim in dim_lst:
            for i in range(20):
                ds = load_dataset(data, uncer_quan_continuous, dim_in=dim, noise_sig2=.01, snr=2, l=1, n_train=n, seed=i)
                x = np.concatenate((ds.x_train, ds.x_test))
                y = np.concatenate((ds.y_train, ds.y_test))
                f = np.concatenate((ds.f_train, ds.f_test))
                df = np.concatenate((f, y, x), axis=1)

                columns = ["f", "y"]
                for j in range(dim):
                    columns.append("x" + str(j + 1))

                df_total = pd.DataFrame(df, columns=columns)
                df_total = df_total.astype(np.float32)
                psi = np.concatenate((ds.psi_train.reshape(1, -1), ds.psi_test.reshape(1, -1)))
                psi = pd.DataFrame(psi)
                df_total.to_csv(
                    "./python/experiments/expr/datasets/cont/{data}_n{n}_d{d}_i{i}.csv".format(data=data, n=n, d=dim, i=i))
                psi.to_csv(
                    "./python/experiments/expr/datasets/cont/psi/{data}_n{n}_d{d}_i{i}.csv".format(data=data, n=n, d=dim, i=i))



# generate adult, heart and mi datasets

def gen_real_data(name):
    for data in data_lst:
        for n in n_lst:
            for dim in dim_lst:
                for i in range(20):
                    ds = load_dataset(data, simu_real, dim_in=dim, noise_sig2=.01, snr=2, l=1, n_train=n, seed=i, base_data = base_data)
                    x_train = np.concatenate(
                        [ds.x_train_cat[:, [0, 1]], ds.x_train[:, [2, 3, 4]], ds.x_train_cat[:, [2, 3]], ds.x_train[:, 7:]],
                        axis=1)
                    x_test = np.concatenate(
                        [ds.x_test_cat[:, [0, 1]], ds.x_test[:, [2, 3, 4]], ds.x_test_cat[:, [2, 3]], ds.x_test[:, 7:]],
                        axis=1)
                    x = np.concatenate((x_train, x_test))
                    while np.sum(np.isnan(x)) > 0:
                        ds = load_dataset(data, simu_real, dim_in=dim, noise_sig2=.01, snr=2, l=1, n_train=n, seed=i, base_data = base_data)
                        x_train = np.concatenate(
                            [ds.x_train_cat[:, [0, 1]], ds.x_train[:, [2, 3, 4]], ds.x_train_cat[:, [2, 3]],
                            ds.x_train[:, 7:]], axis=1)
                        x_test = np.concatenate(
                            [ds.x_test_cat[:, [0, 1]], ds.x_test[:, [2, 3, 4]], ds.x_test_cat[:, [2, 3]], ds.x_test[:, 7:]],
                            axis=1)
                        x = np.concatenate((x_train, x_test))
                    # x = np.concatenate((ds.x_train, ds.x_test))
                    y = np.concatenate((ds.y_train, ds.y_test))
                    f = np.concatenate((ds.f_train, ds.f_test))
                    df = np.concatenate((f, y, x), axis=1)

                    columns = ["f", "y"]
                    for j in range(dim):
                        columns.append("x" + str(j + 1))

                    df_total = pd.DataFrame(df, columns=columns)
                    df_total = df_total.astype(np.float32)
                    # psi = np.concatenate((ds.psi_train.reshape(1, -1), ds.psi_test.reshape(1, -1)))
                    # psi = pd.DataFrame(psi)
                    print(df_total.isnull().values.any())
                    df_total.to_csv(
                        "./python/experiments/expr/datasets/{name}/{data}_n{n}_d{d}_i{i}.csv".format(name=name,
                            data=data, n=n, d=dim, i=i))

# adult dataset
features = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
            "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
            "hours_per_week", "native_country", "label"]

original_train = pd.read_csv("./python/experiments/simu_real/adult/adult.data", names=features,
                             sep=r'\s*, \s*', engine="python", na_values="?")
original_test = pd.read_csv("./python/experiments/simu_real/adult/adult.test", names=features,
                            sep=r'\s*, \s*', engine="python", na_values="?", skiprows=1)
num_train = len(original_train)
original = pd.concat([original_train, original_test])
roc_original = original
labels = original["label"]
labels = labels.replace("<=50K", 0).replace(">50K", 1)
labels = labels.replace("<=50K.", 0).replace(">50K.", 1)

# drop nan
original = original.dropna()

# Redundant column
del original["education"]

# Remove target variable
del original["label"]


adult = original.copy()
# code: White=0, Others=1
adult.loc[~adult["race"].isin(["White"]), "race"] = 1
adult["race"].replace({"White": 0}, inplace=True)

# code: Female=1, Male=0
adult["sex"].replace({"Male": 0, "Female": 1}, inplace=True)

# code: Husband=0, Not-in-family=1, Others=2
adult.loc[~adult["relationship"].isin(["Husband", "Not-in-family"]), "relationship"] = 2
adult["relationship"].replace({"Husband": 0, "Not-in-family": 1}, inplace=True)

# code: Private=0, Self-emp-not-inc=1, Others=2
adult.loc[~adult["workclass"].isin(["Private", "Self-emp-not-inc"]), "workclass"] = 2
adult["workclass"].replace({"Private": 0, "Self-emp-not-inc": 1}, inplace=True)

# code: Married-civ-spouse=0, Never-married=1, Others=2
adult.loc[~adult["marital_status"].isin(["Married-civ-spouse", "Never-married"]), "marital_status"] = 2
adult["marital_status"].replace({"Married-civ-spouse": 0, "Never-married": 1}, inplace=True)

# code: Prof-specialty=0, Craft-repair=1, Others=2
adult.loc[~adult["occupation"].isin(["Prof-specialty", "Craft-repair"]), "occupation"] = 2
adult["occupation"].replace({"Prof-specialty": 0, "Craft-repair": 1}, inplace=True)

# code: United-States=0, Mexico=1, Others=2
adult.loc[~adult["native_country"].isin(["United-States", "Mexico"]), "native_country"] = 2
adult["native_country"].replace({"United-States": 0, "Mexico": 1}, inplace=True)

# check whether there's any missing values
adult.isnull().values.any()

from matplotlib.patches import Rectangle
# create a new dataframe by extracting columns we need, x1-x10
# the categorical covariates are not balanced
feature_mat = adult[["race", "sex", "education_num", "hours_per_week", "age", "relationship",
                     "workclass", "fnlwgt", "capital_gain", "capital_loss", "marital_status",
                     "occupation", "native_country"]].copy()

corrMat = feature_mat.corr().round(2)
ax = sn.heatmap(corrMat, annot=True, cmap='rocket_r')
ax.add_patch(Rectangle((0,0), 5, 5, fill=False, edgecolor='k', lw=3, clip_on=False))
plt.tight_layout()
plt.show()

data_lst = ["linear", "rbf", "matern32", "complex"]
n_lst = [100, 200, 500, 1000]
dim_lst = [25, 50, 100, 200]
base_data = feature_mat.to_numpy()
gen_real_data("adult")

# heart disease data
features = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
            "exang", "oldpeak", "slope", "ca", "thal", "target"]

original = pd.read_csv("./python/experiments/simu_real/heart/processed.cleveland.data", names=features,
                       sep=",", engine="python", na_values="?")

# drop nan
original = original.dropna()

# Remove target variable
del original["target"]

feature_mat = original[["sex", "exang", "thal", "oldpeak", "age", "ca", "cp", "chol",
                        "trestbps", "thalach", "fbs", "restecg", "slope"]].copy()

corrMat = feature_mat.corr().round(2)
ax = sn.heatmap(corrMat, annot=True, cmap='rocket_r')
ax.add_patch(Rectangle((0,0), 5, 5, fill=False, edgecolor='k', lw=3, clip_on=False))
plt.tight_layout()
plt.show()

n_lst = [50, 100, 150, 257]
dim_lst = [25, 50, 100, 200]
base_data = feature_mat.to_numpy()

gen_real_data("heart")

# impute missingness
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
features = ["sex", "ritm_ecg_p_01", "age", "s_ad_orit", "d_ad_orit", "ant_im", "ibs_post", "k_blood",
            "na_blood", "l_blood", "inf_anam", "stenok_an", "fk_stenok", "ibs_nasl", "gb", "sim_gipert",
            "dlit_ag", "zsn_a", "nr11", "nr01", "nr02", "nr03", "nr04", "nr07", "nr08", "np01", "np04",
            "np05", "np07", "np08", "np09", "np10", "endocr_01", "endocr_02", "endocr_03", "zab_leg_01",
            "zab_leg_02", "zab_leg_03", "zab_leg_04", "zab_leg_06", "s_ad_kbrig", "d_ad_kbrig", "o_l_post",
            "k_sh_post", "mp_tp_post", "svt_post", "gt_post", "fib_g_post", "lat_im", "inf_im", "post_im",
            "im_pg_p", "ritm_ecg_p_02", "ritm_ecg_p_04", "ritm_ecg_p_06", "ritm_ecg_p_07", "ritm_ecg_p_08",
            "n_r_ecg_p_01", "n_r_ecg_p_02", "n_r_ecg_p_03", "n_r_ecg_p_04", "n_r_ecg_p_05", "n_r_ecg_p_06",
            "n_r_ecg_p_08", "n_r_ecg_p_09", "n_r_ecg_p_10", "n_p_ecg_p_01", "n_p_ecg_p_03", "n_p_ecg_p_04",
            "n_p_ecg_p_05", "n_p_ecg_p_06", "n_p_ecg_p_07", "n_p_ecg_p_08", "n_p_ecg_p_09", "n_p_ecg_p_10",
            "n_p_ecg_p_11", "n_p_ecg_p_12", "fibr_ter_01", "fibr_ter_02", "fibr_ter_03", "fibr_ter_05",
            "fibr_ter_06", "fibr_ter_07", "fibr_ter_08", "gipo_k", "giper_na", "alt_blood", "ast_blood",
            "kfk_blood", "roe", "time_b_s", "r_ab_1_n", "r_ab_2_n", "r_ab_3_n", "na_kb", "not_na_kb", "lid_kb",
            "nitr_s", "na_r_1_n", "na_r_2_n", "na_r_3_n", "not_na_1_n", "not_na_2_n", "not_na_3_n", "lid_s_n",
            "b_block_s_n", "ant_ca_s_n", "gepar_s_n", "asp_s_n", "tikl_s_n", "trent_s_n"]
feature_mat = original[[2, 49, 1, 36, 37, 44, 6, 83, 85, 89, *list(range(3,6)),
                        *list(range(7,36)), *list(range(38,44)), *list(range(45,49)),
                        *list(range(50,83)), 84, 86, 87, 88, *list(range(90,112))]].copy()
df = feature_mat.set_axis(features, axis=1, inplace=False)
imp = IterativeImputer(random_state=0)
feature_mat = imp.fit_transform(df)
feature_mat = pd.DataFrame(feature_mat, columns=features)
feature_mat.to_csv("./python/experiments/simu_real/mi/feature_processed.csv",
                   columns=features, index=False)

# Myocardial infarction complications Data Set with imputed values
feature_mat = pd.read_csv("./python/experiments/simu_real/mi/feature_processed.csv")
feature_mat = feature_mat.iloc[:, :20]
corrMat = feature_mat.corr().round(2)
ax = sn.heatmap(corrMat, annot=True, cmap='rocket_r', annot_kws={'size':7})
ax.add_patch(Rectangle((0,0), 5, 5, fill=False, edgecolor='k', lw=3, clip_on=False))
plt.tight_layout()
plt.show()

# gen_real_data("mi")
