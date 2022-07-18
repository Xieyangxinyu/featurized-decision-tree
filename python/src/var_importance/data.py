# standard library imports
import os
import sys

# package imports
import autograd.numpy as np
from autograd import grad
# import numpy as np
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
        A matrix D of shape (M, N).  Each entry in D i,j represnets the
        distance between row i in A and row j in B.
    """
    M = A.shape[0]
    N = B.shape[0]

    assert A.shape[1] == B.shape[1], f"The number of components for vectors in A \
        {A.shape[1]} does not match that of B {B.shape[1]}!"

    A_dots = (A*A).sum(axis=1).reshape((M,1))*np.ones(shape=(1,N))
    B_dots = (B*B).sum(axis=1)*np.ones(shape=(M,1))
    D_squared =  A_dots + B_dots -2*np.dot(A, B.T)

    if squared == False:
        D_squared = np.where(D_squared > 0.0, D_squared, 0.0)
        return np.sqrt(D_squared)

    return D_squared


class Dataset:
    x_train = None # inputs
    z_train = None # covariates
    f_train = None # ground truth function
    y_train = None # observed outcomes
    psi_train = None # variable importance

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

        self.f = lambda x: f(x).reshape(-1,1) # makes sure output is (n,1)
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
        self.x_train_cat = x_train[:, [0, 1, 7, 8]]
        self.x_test_cat = x_test[:, [0, 1, 7, 8]]

        self.evaluate_f()
        self.sample_y(seed)

        self.standardized = False
        if standardize:
            self.standardize()
        try:
            self.evaluate_psi() # note: psi always based on whether data originally standardized
        except:
            print('Unable to compute variable importance, possible autograd not imported')

    def evaluate_f(self):

        scale_factor = self.snr * np.sqrt(self.noise_sig2)
        # train
        self.f_train = self.f(self.x_train).reshape(-1,1)
        scaler = preprocessing.StandardScaler(with_mean=False).fit(self.f_train)
        f_train_scaled = scaler.transform(self.f_train)
        self.f_train = scale_factor * f_train_scaled

        # test
        if self.x_test is not None:
            self.f_test = self.f(self.x_test).reshape(-1,1)
            scaler = preprocessing.StandardScaler(with_mean=False).fit(self.f_test)
            f_test_scaled = scaler.transform(self.f_test)
            self.f_test = scale_factor * f_test_scaled

    def sample_y(self, seed=0):
        r_noise = np.random.RandomState(seed)
        
        # train
        noise = r_noise.randn(self.n_train,1) * np.sqrt(self.noise_sig2)
        self.y_train = self.f_train + noise

        # test
        if self.x_test is not None:
            noise_test = r_noise.randn(self.n_test,1) * np.sqrt(self.noise_sig2)
            self.y_test = self.f_test + noise_test

    def evaluate_psi(self):
        grad_f = grad(self.f)
        jac_f_train = np.zeros((self.n_train, self.dim_in))
        for i in range(self.n_train):
            jac_f_train[i,:] = grad_f(self.x_train[i,:].reshape(1,-1))
        self.psi_train = np.mean(jac_f_train**2, 0)

        if self.x_test is not None:
            jac_f_test = np.zeros((self.n_test, self.dim_in))
            for i in range(self.n_test):
                jac_f_test[i, :] = grad_f(self.x_test[i, :].reshape(1,-1))
            self.psi_test = np.mean(jac_f_test ** 2, 0)

    def standardize(self):
        zscore = lambda x, mu, sigma: (x - mu.reshape(1,-1)) / sigma.reshape(1,-1)
        un_zscore = lambda x, mu, sigma: x * sigma.reshape(1,-1) + mu.reshape(1,-1)
            
        if not self.standardized:
            
            self.mu_x = np.mean(self.x_train, axis=0)
            self.sigma_x = np.std(self.x_train, axis=0)

            #self.mu_f = np.mean(self.f_train, axis=0)
            #self.sigma_f = np.std(self.f_train, axis=0)

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
        zscore = lambda x, mu, sigma: (x - mu.reshape(1,-1)) / sigma.reshape(1,-1)
        un_zscore = lambda x, mu, sigma: x * sigma.reshape(1,-1) + mu.reshape(1,-1)    

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


# explicitly generate interaction, categorical data
def uncer_quan(dim_in, noise_sig2, snr, l, n_train, n_test=40, seed_x=0, seed_noise=0, version=1):

    # sample x
    r_x = np.random.RandomState(seed_x)
    n_all = n_train + n_test
    x_all = r_x.uniform(-2, 2, (n_all, dim_in))
    x_all[:, [0, 1, 5, 6]] = r_x.binomial(1, 0.5, size=n_all*4).reshape((n_all, 4))

    tmp = np.insert(x_all, 5, x_all[:,0] * x_all[:,2], axis=1)
    tmp = np.insert(tmp, 6, 0.3 * x_all[:, 0] * x_all[:, 3], axis=1)
    x_all = tmp
    x_train = x_all[:n_train, :]
    x_test = x_all[n_train:, :]

    if version==1:
        assert dim_in >= 5
        f = lambda x: x[:,0] - x[:,1] + x[:,2] + 0.5*x[:,3] + 2*x[:,4] + x[:,5] + x[:,6]

    elif version==2:
        assert dim_in >= 5
        r_f = np.random.RandomState(seed_x)
        # sample 5D function
        n_obs_sample = 500
        x0 = r_f.uniform(-2, 2, (n_obs_sample, 5))
        x0[:, [0, 1]] = r_f.binomial(1, 0.5, size=n_obs_sample * 2).reshape((n_obs_sample, 2))
        x0 = np.append(x0, (x0[:, 0] * x0[:, 2]).reshape((n_obs_sample, 1)), axis=1)
        x0 = np.append(x0, (0.3*(x0[:, 0] * x0[:, 3])).reshape((n_obs_sample, 1)), axis=1)
        act = lambda z: np.exp(-distance_matrix(z, x0) ** 2 / (2 * l**2))
        rbf_kern = act(x0)
        f0 = r_f.multivariate_normal(np.zeros(n_obs_sample), rbf_kern, 1).reshape(-1, 1)
        y0 = f0 + r_f.normal(0, np.sqrt(noise_sig2), (n_obs_sample, 1))
        alpha = np.linalg.inv(rbf_kern + noise_sig2*np.identity(x0.shape[0]))@y0
        f = lambda z: act(z[:, :7])@alpha

    elif version==3:
        assert dim_in >= 5
        r_f = np.random.RandomState(seed_x)
        # sample 5D function
        n_obs_sample = 500
        x0 = r_f.uniform(-2, 2, (n_obs_sample, 5))
        x0[:, [0, 1]] = r_f.binomial(1, 0.5, size=n_obs_sample * 2).reshape((n_obs_sample, 2))
        x0 = np.append(x0, (x0[:, 0] * x0[:, 2]).reshape((n_obs_sample, 1)), axis=1)
        x0 = np.append(x0, (0.3 * (x0[:, 0] * x0[:, 3])).reshape((n_obs_sample, 1)), axis=1)
        act = lambda z: (1 + np.sqrt(3) * distance_matrix(z, x0) / l) * np.exp(-np.sqrt(3) * distance_matrix(z, x0) / l)
        mat32_kern = act(x0)
        f0 = r_f.multivariate_normal(np.zeros(n_obs_sample), mat32_kern, 1).reshape(-1, 1)
        y0 = f0 + r_f.normal(0, np.sqrt(noise_sig2), (n_obs_sample, 1))
        alpha = np.linalg.inv(mat32_kern + noise_sig2*np.identity(x0.shape[0]))@y0
        f = lambda z: act(z[:, :7])@alpha

    elif version==4:
        assert dim_in >= 5
        f = lambda x: np.sin(np.max([x[:,0], x[:,1]]) + np.arctan(x[:,1])) / (1+x[:,0]+x[:,4]) + \
                      np.sin(0.5*x[:,2]) * (1+np.exp(x[:,3] - 0.5*x[:,2])) + x[:,2]**2 + \
                      2*np.sin(x[:,3]) + 4*x[:,4] + np.sin(x[:, 0] * x[:, 2] + 0.3 * (x[:, 0] * x[:, 3]))

    return Toy(f, x_train, x_test, noise_sig2, snr, l, seed_noise)

## original code, continuous data
def uncer_quan(dim_in, noise_sig2, snr, l, n_train, n_test=40, seed_x=0, seed_noise=0, version=1):

    # sample x
    r_x = np.random.RandomState(seed_x)
    n_all = n_train + n_test
    x_all = r_x.uniform(-2, 2, (n_all, dim_in))

    tmp = np.insert(x_all, 5, x_all[:, 0] * x_all[:, 2], axis=1)
    tmp = np.insert(tmp, 6, 0.3 * x_all[:, 0] * x_all[:, 3], axis=1)
    x_all = tmp
    x_train = x_all[:n_train, :]
    x_test = x_all[n_train:, :]

    if version==1:
        assert dim_in >= 5
        f = lambda x: x[:,0] - x[:,1] + x[:,2] + 0.5*x[:,3] + 2*x[:,4] + x[:,5] + x[:,6]

    elif version==2:
        assert dim_in >= 5
        r_f = np.random.RandomState(seed_x)
        # sample 5D function
        n_obs_sample = 500
        x0 = r_f.uniform(-2, 2, (n_obs_sample, 5))
        x0 = np.append(x0, (x0[:, 0] * x0[:, 2]).reshape((n_obs_sample, 1)), axis=1)
        x0 = np.append(x0, (0.3 * (x0[:, 0] * x0[:, 3])).reshape((n_obs_sample, 1)), axis=1)
        act = lambda z: np.exp(-distance_matrix(z, x0) ** 2 / (2 * l**2))
        rbf_kern = act(x0)
        f0 = r_f.multivariate_normal(np.zeros(n_obs_sample), rbf_kern, 1).reshape(-1, 1)
        y0 = f0 + r_f.normal(0, np.sqrt(noise_sig2), (n_obs_sample, 1))
        alpha = np.linalg.inv(rbf_kern + noise_sig2*np.identity(x0.shape[0]))@y0
        f = lambda z: act(z[:, :7])@alpha

    elif version==3:
        assert dim_in >= 5
        r_f = np.random.RandomState(seed_x)
        # sample 5D function
        n_obs_sample = 500
        x0 = r_f.uniform(-2, 2, (n_obs_sample, 5))
        x0 = np.append(x0, (x0[:, 0] * x0[:, 2]).reshape((n_obs_sample, 1)), axis=1)
        x0 = np.append(x0, (0.3 * (x0[:, 0] * x0[:, 3])).reshape((n_obs_sample, 1)), axis=1)
        act = lambda z: (1 + np.sqrt(3) * distance_matrix(z, x0) / l) * np.exp(-np.sqrt(3) * distance_matrix(z, x0) / l)
        mat32_kern = act(x0)
        f0 = r_f.multivariate_normal(np.zeros(n_obs_sample), mat32_kern, 1).reshape(-1, 1)
        y0 = f0 + r_f.normal(0, np.sqrt(noise_sig2), (n_obs_sample, 1))
        alpha = np.linalg.inv(mat32_kern + noise_sig2*np.identity(x0.shape[0]))@y0
        f = lambda z: act(z[:, :7])@alpha

    elif version==4:
        assert dim_in >= 5
        f = lambda x: np.sin(np.max([x[:,0], x[:,1]]) + np.arctan(x[:,1])) / (1+x[:,0]+x[:,4]) + \
                      np.sin(0.5*x[:,2]) * (1+np.exp(x[:,3] - 0.5*x[:,2])) + x[:,2]**2 + \
                      2*np.sin(x[:,3]) + 4*x[:,4] + np.sin(x[:, 0] * x[:, 2] + 0.3 * (x[:, 0] * x[:, 3]))

    return Toy(f, x_train, x_test, noise_sig2, snr, l, seed_noise)


#
# ## 1-d prediction
# def uncer_quan(dim_in, noise_sig2, snr, l, n_train, n_test=50, seed_x=0, seed_noise=0, version=1):
#
#     # sample x
#     r_x = np.random.RandomState(seed_x)
#     n_all = n_train + n_test
#     x_all = r_x.uniform(-2, 2, (n_all, dim_in))
#     x_train = x_all[:n_train, :]
#     x_test = x_all[n_train:, :]
#
#     if version==1:
#         f = lambda x: x[:,0]
#
#     elif version==2:
#         r_f = np.random.RandomState(seed_x)
#         # sample 5D function
#         n_obs_sample = 500
#         x0 = r_f.uniform(-2, 2, (n_obs_sample, 1))
#         act = lambda z: np.exp(-distance_matrix(z, x0) ** 2 / (2 * l**2))
#         rbf_kern = act(x0)
#         f0 = r_f.multivariate_normal(np.zeros(n_obs_sample), rbf_kern, 1).reshape(-1, 1)
#         y0 = f0 + r_f.normal(0, np.sqrt(noise_sig2), (n_obs_sample, 1))
#         alpha = np.linalg.inv(rbf_kern + noise_sig2*np.identity(x0.shape[0]))@y0
#         f = lambda z: act(z[:, 0].reshape((-1,1)))@alpha
#
#     elif version==3:
#         r_f = np.random.RandomState(seed_x)
#         # sample 5D function
#         n_obs_sample = 500
#         x0 = r_f.uniform(-2, 2, (n_obs_sample, 1))
#         act = lambda z: (1 + np.sqrt(3) * distance_matrix(z, x0) / l) * np.exp(-np.sqrt(3) * distance_matrix(z, x0) / l)
#         mat32_kern = act(x0)
#         f0 = r_f.multivariate_normal(np.zeros(n_obs_sample), mat32_kern, 1).reshape(-1, 1)
#         y0 = f0 + r_f.normal(0, np.sqrt(noise_sig2), (n_obs_sample, 1))
#         alpha = np.linalg.inv(mat32_kern + noise_sig2*np.identity(x0.shape[0]))@y0
#         f = lambda z: act(z[:, 0].reshape((-1,1)))@alpha
#
#     elif version==4:
#         f = lambda x: np.sin(np.arctan(x[:, 0])) / (1 + x[:, 0]) + \
#                       np.sin(0.5 * x[:, 0]) * (1 + np.exp(0.5 * x[:, 0])) + x[:, 0] ** 2 + \
#                       2 * np.sin(x[:, 0]) + 4 * x[:, 0] + np.sin(x[:, 0] * x[:, 2] + 0.3 * (x[:, 0] * x[:, 3]))
#
#     return Toy(f, x_train, x_test, noise_sig2, snr, l, seed_noise)



# simulation using features from real datasets
def simu_real(base_data, dim_in, noise_sig2, snr, l, n_train, n_test=50, seed_x=0, seed_noise=0, version=1):

    # sample x
    dim_original = base_data.shape[1]
    r_x = np.random.RandomState(seed_x)
    n_all = n_train + n_test
    x_original = base_data[np.random.choice(base_data.shape[0], size=n_all, replace=False), :]
    x_rest = r_x.uniform(-2, 2, (n_all, dim_in - dim_original))
    x_all = np.concatenate((x_original, x_rest), axis=1)

    tmp = np.insert(x_all, 5, x_all[:,0] * x_all[:,2], axis=1)
    tmp = np.insert(tmp, 6, 0.3 * x_all[:, 0] * x_all[:, 3], axis=1)
    x_all = tmp
    x_train = x_all[:n_train, :]
    x_test = x_all[n_train:, :]

    if version==1:
        assert dim_in >= 5
        f = lambda x: x[:,0] - x[:,1] + x[:,2] + 0.5*x[:,3] + 2*x[:,4] + x[:,5] + x[:,6]

    elif version==2:
        assert dim_in >= 5
        r_f = np.random.RandomState(seed_x)
        # sample 5D function
        n_obs_sample = 500
        x0 = r_f.uniform(-2, 2, (n_obs_sample, 5))
        x0[:, [0, 1]] = r_f.binomial(1, 0.5, size=n_obs_sample * 2).reshape((n_obs_sample, 2))
        x0 = np.append(x0, (x0[:, 0] * x0[:, 2]).reshape((n_obs_sample, 1)), axis=1)
        x0 = np.append(x0, (0.3*(x0[:, 0] * x0[:, 3])).reshape((n_obs_sample, 1)), axis=1)
        act = lambda z: np.exp(-distance_matrix(z, x0) ** 2 / (2 * l**2))
        rbf_kern = act(x0)
        f0 = r_f.multivariate_normal(np.zeros(n_obs_sample), rbf_kern, 1).reshape(-1, 1)
        y0 = f0 + r_f.normal(0, np.sqrt(noise_sig2), (n_obs_sample, 1))
        alpha = np.linalg.inv(rbf_kern + noise_sig2*np.identity(x0.shape[0]))@y0
        f = lambda z: act(z[:, :7])@alpha

    elif version==3:
        assert dim_in >= 5
        r_f = np.random.RandomState(seed_x)
        # sample 5D function
        n_obs_sample = 500
        x0 = r_f.uniform(-2, 2, (n_obs_sample, 5))
        x0[:, [0, 1]] = r_f.binomial(1, 0.5, size=n_obs_sample * 2).reshape((n_obs_sample, 2))
        x0 = np.append(x0, (x0[:, 0] * x0[:, 2]).reshape((n_obs_sample, 1)), axis=1)
        x0 = np.append(x0, (0.3 * (x0[:, 0] * x0[:, 3])).reshape((n_obs_sample, 1)), axis=1)
        act = lambda z: (1 + np.sqrt(3) * distance_matrix(z, x0) / l) * np.exp(-np.sqrt(3) * distance_matrix(z, x0) / l)
        mat32_kern = act(x0)
        f0 = r_f.multivariate_normal(np.zeros(n_obs_sample), mat32_kern, 1).reshape(-1, 1)
        y0 = f0 + r_f.normal(0, np.sqrt(noise_sig2), (n_obs_sample, 1))
        alpha = np.linalg.inv(mat32_kern + noise_sig2*np.identity(x0.shape[0]))@y0
        f = lambda z: act(z[:, :7])@alpha

    elif version==4:
        assert dim_in >= 5
        f = lambda x: np.sin(np.max([x[:,0], x[:,1]]) + np.arctan(x[:,1])) / (1+x[:,0]+x[:,4]) + \
                      np.sin(0.5*x[:,2]) * (1+np.exp(x[:,3] - 0.5*x[:,2])) + x[:,2]**2 + \
                      2*np.sin(x[:,3]) + 4*x[:,4] + np.sin(x[:, 0] * x[:, 2] + 0.3 * (x[:, 0] * x[:, 3]))

    return Toy(f, x_train, x_test, noise_sig2, snr, l, seed_noise)

def load_dataset(name, dim_in, noise_sig2, snr, l, n_train, n_test=40, seed=0):
    '''
    inputs:

    returns:
    '''

    if name == 'linear':
        dataset = uncer_quan(dim_in, noise_sig2, snr, l, n_train, n_test=n_test, seed_x=seed, seed_noise=seed, version=1)

    elif name == 'rbf':
        dataset = uncer_quan(dim_in, noise_sig2, snr, l, n_train, n_test=n_test, seed_x=seed, seed_noise=seed, version=2)

    elif name == 'matern32':
        dataset = uncer_quan(dim_in, noise_sig2, snr, l, n_train, n_test=n_test, seed_x=seed, seed_noise=seed, version=3)

    elif name == 'complex':
        dataset = uncer_quan(dim_in, noise_sig2, snr, l, n_train, n_test=n_test, seed_x=seed, seed_noise=seed, version=4)

    return dataset

def load_synthetic_dataset(base_data, name, dim_in, noise_sig2, snr, l, n_train, n_test=50, seed=0):
    '''
    inputs:

    returns:
    '''

    if name == 'linear':
        dataset = simu_real(base_data, dim_in, noise_sig2, snr, l, n_train, n_test=n_test, seed_x=seed, seed_noise=seed, version=1)

    elif name == 'rbf':
        dataset = simu_real(base_data, dim_in, noise_sig2, snr, l, n_train, n_test=n_test, seed_x=seed, seed_noise=seed, version=2)

    elif name == 'matern32':
        dataset = simu_real(base_data, dim_in, noise_sig2, snr, l, n_train, n_test=n_test, seed_x=seed, seed_noise=seed, version=3)

    elif name == 'complex':
        dataset = simu_real(base_data, dim_in, noise_sig2, snr, l, n_train, n_test=n_test, seed_x=seed, seed_noise=seed, version=4)

    return dataset



data_lst = ["linear", "rbf", "matern32", "complex"]
n_lst = [100, 200, 500, 1000]
dim_lst = [25, 50, 100]

for data in data_lst:
    for n in n_lst:
        for dim in dim_lst:
            for i in range(20):
                ds = load_dataset(data, dim_in=dim, noise_sig2=.01, snr=2, l=1, n_train=n, seed=i)
                x_train = np.concatenate([ds.x_train_cat[:,[0,1]], ds.x_train[:,[2,3,4]], ds.x_train_cat[:,[2,3]], ds.x_train[:,9:]], axis=1)
                x_test = np.concatenate([ds.x_test_cat[:,[0,1]], ds.x_test[:,[2,3,4]], ds.x_test_cat[:,[2,3]], ds.x_test[:,9:]], axis=1)
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
                psi = np.concatenate((ds.psi_train.reshape(1,-1), ds.psi_test.reshape(1,-1)))
                psi = pd.DataFrame(psi)
                df_total.to_csv("/Users/wdeng/Desktop/data_cat/{data}_n{n}_d{d}_i{i}.csv".format(data=data, n=n, d=dim, i=i))
                psi.to_csv("/Users/wdeng/Desktop/data_cat/psi/{data}_n{n}_d{d}_i{i}.csv".format(data=data, n=n, d=dim, i=i))



for data in data_lst:
    for n in n_lst:
        for dim in dim_lst:
            for i in range(20):
                ds = load_dataset(data, dim_in=dim, noise_sig2=.01, snr=2, l=1, n_train=n, seed=i)
                x_train = np.concatenate([ds.x_train[:, 0:5], ds.x_train[:, 7:]], axis=1)
                x_test = np.concatenate([ds.x_test[:, 0:5], ds.x_test[:, 7:]], axis=1)
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
                psi = np.concatenate((ds.psi_train.reshape(1,-1), ds.psi_test.reshape(1,-1)))
                psi = pd.DataFrame(psi)
                df_total.to_csv("/Users/wdeng/Desktop/data_cont/{data}_n{n}_d{d}_i{i}.csv".format(data=data, n=n, d=dim, i=i))
                psi.to_csv("/Users/wdeng/Desktop/data_cont/psi/{data}_n{n}_d{d}_i{i}.csv".format(data=data, n=n, d=dim, i=i))


# categorical and continuous data
# n_lst = [100, 200, 500, 1000]
# heart
# n_lst = [50, 100, 150, 257]
data_lst = ["linear", "rbf", "matern32", "complex"]
n_lst = [100, 200, 500, 1000]
dim_lst = [25, 50, 100]
# dd = feature_mat.iloc[:, range(100)]
# base_data = dd.to_numpy()
base_data = feature_mat.to_numpy()
for data in data_lst:
    for n in n_lst:
        for dim in dim_lst:
            for i in range(20):
                ds = load_synthetic_dataset(base_data, data, dim_in=dim, noise_sig2=.01, snr=2, l=1, n_train=n, n_test=40, seed=i)
                x_train = np.concatenate([ds.x_train_cat[:,[0,1]], ds.x_train[:,[2,3,4]], ds.x_train_cat[:,[2,3]], ds.x_train[:,9:]], axis=1)
                x_test = np.concatenate([ds.x_test_cat[:,[0,1]], ds.x_test[:,[2,3,4]], ds.x_test_cat[:,[2,3]], ds.x_test[:,9:]], axis=1)
                x = np.concatenate((x_train, x_test))
                while np.sum(np.isnan(x)) > 0:
                    ds = load_synthetic_dataset(base_data, data, dim_in=dim, noise_sig2=.01, snr=2, l=1, n_train=n,
                                                n_test=40, seed=i)
                    x_train = np.concatenate(
                        [ds.x_train_cat[:, [0, 1]], ds.x_train[:, [2, 3, 4]], ds.x_train_cat[:, [2, 3]],
                         ds.x_train[:, 9:]], axis=1)
                    x_test = np.concatenate(
                        [ds.x_test_cat[:, [0, 1]], ds.x_test[:, [2, 3, 4]], ds.x_test_cat[:, [2, 3]], ds.x_test[:, 9:]],
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
                psi = np.concatenate((ds.psi_train.reshape(1,-1), ds.psi_test.reshape(1,-1)))
                psi = pd.DataFrame(psi)
                print(df_total.isnull().values.any())
                df_total.to_csv("/Users/wdeng/Desktop/FDT/FDT/python/experiments/expr/datasets/mi/{data}_n{n}_d{d}_i{i}.csv".format(data=data, n=n, d=dim, i=i))
                psi.to_csv("/Users/wdeng/Desktop/FDT/FDT/python/experiments/expr/datasets/mi/psi/{data}_n{n}_d{d}_i{i}.csv".format(data=data, n=n, d=dim, i=i))


# continuous data
data_lst = ["linear", "rbf", "matern32", "complex"]
n_lst = [100, 200, 500, 1000]
dim_lst = [25, 50, 100, 200]
base_data = feature_mat.to_numpy()
for data in data_lst:
    for n in n_lst:
        for dim in dim_lst:
            for i in range(20):
                ds = load_synthetic_dataset(base_data, data, dim_in=dim, noise_sig2=.01, snr=2, l=1, n_train=n, n_test=40, seed=i)
                x_train = np.concatenate([ds.x_train[:,0:5], ds.x_train[:,7:]], axis=1)
                x_test = np.concatenate([ds.x_test[:,0:5], ds.x_test[:,7:]], axis=1)
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
                psi = np.concatenate((ds.psi_train.reshape(1,-1), ds.psi_test.reshape(1,-1)))
                psi = pd.DataFrame(psi)
                df_total.to_csv("/Users/irisdeng/Desktop/simu_data/pca/adult/{data}_n{n}_d{d}_i{i}.csv".format(data=data, n=n, d=dim, i=i))
                psi.to_csv("/Users/irisdeng/Desktop/simu_data/pca/adult/psi/{data}_n{n}_d{d}_i{i}.csv".format(data=data, n=n, d=dim, i=i))

