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
import numpy as np
import csv
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
import sklearn.preprocessing as preprocessing

import seaborn as sn
import matplotlib.pyplot as plt

# --------- Load data -----------

# process bangladesh original data
data_raw = pd.read_csv("/Users/irisdeng/Downloads/bangladesh/data_stand.csv", index_col=0)
data = data_raw[['clinic', 'examiner', 'sex', 'deliv_type', 'cognitive', 'gest_weeks',
                 'birth_order', 'drinkwater_cups', 'hosp_child_yn', 'pica_yn', 'education',
                 'educa_spouse', 'smokenv', 'home_emo', 'home_avoid', 'home_careg',
                 'home_env', 'home_play', 'home_stim', 'home_clean', 'age_visit',
                 'length', 'weight', 'head_circ', 'sbp', 'dbp', 'blood_as', 'blood_mn',
                 'blood_pb', 'blood_pb_b', 'blood_mn_b', 'blood_as_b', 'energy',
                 'prot', 'fat', 'carb', 'fib', 'ash', 'as_ln', 'mn_ln', 'pb_ln']].copy()
cat_df = data[['clinic', 'examiner', 'sex', 'deliv_type']].copy()
cont_features = ['cognitive', 'gest_weeks',
                 'birth_order', 'drinkwater_cups', 'hosp_child_yn', 'pica_yn', 'education',
                 'educa_spouse', 'smokenv', 'home_emo', 'home_avoid', 'home_careg',
                 'home_env', 'home_play', 'home_stim', 'home_clean', 'age_visit',
                 'length', 'weight', 'head_circ', 'sbp', 'dbp', 'blood_as', 'blood_mn',
                 'blood_pb', 'blood_pb_b', 'blood_mn_b', 'blood_as_b', 'energy',
                 'prot', 'fat', 'carb', 'fib', 'ash', 'as_ln', 'mn_ln', 'pb_ln']
cont_df = data[cont_features].copy()
scaler = preprocessing.StandardScaler()
cont_df = pd.DataFrame(scaler.fit_transform(cont_df), columns = cont_features)
data = pd.concat([cat_df, cont_df], axis=1)
data = data.dropna()
df_train = data.copy()
x_all, y_all = df_train, df_train.pop("cognitive")
X = x_all[['clinic', 'examiner', 'sex', 'deliv_type', 'gest_weeks', 'birth_order',
           'drinkwater_cups', 'hosp_child_yn', 'pica_yn', 'education',
           'educa_spouse', 'smokenv', 'home_emo', 'home_avoid', 'home_careg',
           'home_env', 'home_play', 'home_stim', 'home_clean', 'age_visit',
           'length', 'weight', 'head_circ', 'sbp', 'dbp', 'blood_as', 'blood_mn',
           'blood_pb', 'blood_pb_b', 'blood_mn_b', 'blood_as_b', 'energy']].copy()
X = X.to_numpy()
y_all = y_all.to_numpy().reshape(-1, 1).ravel()

lnr_reg = LinearRegression().fit(X, y_all)
res = y_all - lnr_reg.predict(X)
data_all = pd.concat([pd.DataFrame(res, columns=["residual"]), data.reset_index(drop=True)], axis=1)


# process yacht original data
feature_name = ['Long pos', 'Prismatic coeff', 'Length-displacement ratio', 'Beam-draught ratio',
                'Length-beam ratio', 'Froude number', 'Residuary resistance']
data_raw = pd.read_csv("/Users/irisdeng/Downloads/yacht/yacht_hydrodynamics.data",
                       delim_whitespace=True, names=feature_name)
scaler = preprocessing.StandardScaler()
df_train = pd.DataFrame(scaler.fit_transform(data_raw), columns = feature_name)
df_train = df_train.dropna()
df_train.to_csv("/Users/irisdeng/Downloads/yacht/data_processed.csv", index=False)
# don't need to perform linear regression first to get residuals because we later
# do nonlinear regression using all covariates
Z, res = df_train, df_train.pop("Residuary resistance")
Z = Z.to_numpy()
n_obs, dim_in = Z.shape
res = res.to_numpy().reshape(-1, 1).ravel()
Z_train = Z[:258, :]
Z_test = Z[258:, :]
y_train = res[:258]
y_test = res[258:]
n_train = 258


# process concrete original data
feature_name = ['Cement', 'Blast Furnace Slag', 'Fly Ash', 'Water', 'Superplasticizer',
                'Coarse Aggregate', 'Fine Aggregate', 'Age', 'Concrete compressive strength']
data_raw = pd.read_excel("/Users/irisdeng/Downloads/concrete/Concrete_Data.xls", names=feature_name)
scaler = preprocessing.StandardScaler()
df_train = pd.DataFrame(scaler.fit_transform(data_raw), columns = feature_name)
df_train = df_train.dropna()
df_train.to_csv("/Users/irisdeng/Downloads/concrete/data_processed.csv", index=False)
# don't need to perform linear regression first to get residuals because we later
# do nonlinear regression using all covariates
Z, res = df_train, df_train.pop("Concrete compressive strength")
Z = Z.to_numpy()
n_obs, dim_in = Z.shape
res = res.to_numpy().reshape(-1, 1).ravel()
Z_train = Z[:980, :]
Z_test = Z[980:, :]
y_train = res[:980]
y_test = res[980:]
n_train = 980


## find the best sig2
df_train = pd.read_csv("/Users/irisdeng/Downloads/real_data/concrete/data_processed.csv")
x_all, y_all, res = df_train, df_train.pop("cognitive"), df_train.pop("residual")
X = x_all[['clinic', 'examiner', 'sex', 'deliv_type', 'gest_weeks', 'birth_order',
           'drinkwater_cups', 'hosp_child_yn', 'pica_yn', 'education',
           'educa_spouse', 'smokenv', 'home_emo', 'home_avoid', 'home_careg',
           'home_env', 'home_play', 'home_stim', 'home_clean', 'age_visit',
           'length', 'weight', 'head_circ', 'sbp', 'dbp', 'blood_as', 'blood_mn',
           'blood_pb', 'blood_pb_b', 'blood_mn_b', 'blood_as_b', 'energy']].copy()
Z = x_all[['clinic', 'sex', 'prot', 'fat', 'carb', 'fib', 'ash', 'as_ln', 'mn_ln', 'pb_ln']].copy()
X = X.to_numpy()
Z = Z.to_numpy()
dim_in = Z.shape[1]
res = res.to_numpy().reshape(-1, 1).ravel()
y_all = y_all.to_numpy().reshape(-1, 1).ravel()
Z_train = Z[:666, :]
Z_test = Z[666:, :]
y_train = res[:666]
y_test = res[666:]
n_train = 666

max_leaf_nodes = 2**(int(np.log2(np.sqrt(n_train) * np.log(n_train))) + 1)
rf = ExtraTreesRegressor(n_estimators=20, max_leaf_nodes=max_leaf_nodes, random_state=0).fit(Z_train, y_train)

stable_sigmoid = lambda x: jnp.exp(jax.nn.log_sigmoid(x))
c = 1000
def predict(x, map_matrix, feature, threshold, beta):
    hidden_features = jnp.asarray(x)[feature]
    right_indicator = stable_sigmoid(c * (hidden_features - threshold))
    left_indicator = 1. - right_indicator
    soft_indicator = jnp.concatenate([left_indicator, right_indicator], axis=0)
    F_leaf = jnp.multiply(soft_indicator[:, np.newaxis], map_matrix) + (1.0 - map_matrix)
    soft_feature = jnp.prod(F_leaf, axis=0)
    return jnp.dot(soft_feature, beta)

pred_mse_lst = []
sigma_2_lst = np.e ** np.linspace(-3, 3, 7)
for sigma_2 in sigma_2_lst:
    beta_set = np.zeros((rf.n_estimators, max_leaf_nodes))
    map_matrix_set = np.zeros((rf.n_estimators, 2 * (max_leaf_nodes - 1), max_leaf_nodes))
    feature_set = np.zeros((rf.n_estimators, max_leaf_nodes - 1))
    threshold_set = np.zeros((rf.n_estimators, max_leaf_nodes - 1))
    for j, model in enumerate(rf.estimators_):
        fdt = FDT(model, Z_train, y_train, c=c, sig2=sigma_2)
        fdt.train()
        beta_set[j, :] = fdt.beta
        map_matrix_set[j, :, :] = fdt.map_matrix
        feature_set[j, :] = fdt.hidden_features
        threshold_set[j, :] = fdt.hidden_threshold

    beta_set = jnp.asarray(beta_set)
    map_matrix_set = jnp.asarray(map_matrix_set)
    feature_set = jnp.asarray(feature_set, dtype=int)
    threshold_set = jnp.asarray(threshold_set)

    f = jax.jit(jax.vmap(jax.vmap(predict, in_axes=(None, 0, 0, 0, 0), out_axes=0),
                         in_axes=(0, None, None, None, None), out_axes=0))

    y_test_pred = np.array(f(Z_test, map_matrix_set, feature_set, threshold_set, beta_set))
    y_test_pred = np.mean(y_test_pred, 1)
    tst_mse_fdt = np.mean((y_test - y_test_pred) ** 2)
    pred_mse_lst.append(np.mean((y_test_pred - y_test) ** 2) / np.mean((y_test - np.mean(y_test)) ** 2))

sig2 = sigma_2_lst[np.argmin(pred_mse_lst)]
n_obs = n_train
max_leaf_nodes = 2**(int(np.log2(np.sqrt(n_obs) * np.log(n_obs))) + 1)
rf = ExtraTreesRegressor(n_estimators=50, max_leaf_nodes=max_leaf_nodes, random_state=0).fit(Z, res)
rf_raw = RandomForestRegressor(n_estimators=50, max_leaf_nodes=max_leaf_nodes, random_state=0).fit(Z, res)
beta_set = np.zeros((rf.n_estimators, max_leaf_nodes))
map_matrix_set = np.zeros((rf.n_estimators, 2 * (max_leaf_nodes - 1), max_leaf_nodes))
feature_set = np.zeros((rf.n_estimators, max_leaf_nodes - 1))
threshold_set = np.zeros((rf.n_estimators, max_leaf_nodes - 1))

for j, model in enumerate(rf.estimators_):
    fdt = FDT(model, Z, res, c=c, sig2=sig2)
    fdt.train()
    beta_set[j, :] = fdt.beta
    map_matrix_set[j, :, :] = fdt.map_matrix
    feature_set[j, :] = fdt.hidden_features
    threshold_set[j, :] = fdt.hidden_threshold

beta_set = jnp.asarray(beta_set)
map_matrix_set = jnp.asarray(map_matrix_set)
feature_set = jnp.asarray(feature_set, dtype=int)
threshold_set = jnp.asarray(threshold_set)

f = jax.jit(jax.vmap(jax.vmap(predict, in_axes=(None, 0, 0, 0, 0), out_axes=0),
                         in_axes=(0, None, None, None, None), out_axes=0))

grad_f = jax.jit(jax.vmap(jax.vmap(jax.grad(predict, argnums=0),
                                   in_axes=(None, 0, 0, 0, 0), out_axes=0),
                          in_axes=(0, None, None, None, None), out_axes=0))


def split_into_batches(X, batch_size):
    return [X[i:i + batch_size] for i in range(0, len(X), batch_size)]

Z_batches = split_into_batches(Z, batch_size=100)
psi_est_all = np.array(grad_f(Z_batches[0], map_matrix_set, feature_set, threshold_set, beta_set))

for batch_id in range(1, Z_batches.__len__()):
    Z_batch = Z_batches[batch_id]
    psi_est_tmp = np.array(grad_f(Z_batch, map_matrix_set, feature_set, threshold_set, beta_set))
    psi_est_all = np.concatenate([psi_est_all, psi_est_tmp], axis=0)


Z1 = Z[:340, :]
Z2 = Z[340:680, :]
Z3 = Z[680:, :]
psi_est_all1 = np.array(grad_f(Z1, map_matrix_set, feature_set, threshold_set, beta_set))
psi_est_all2 = np.array(grad_f(Z2, map_matrix_set, feature_set, threshold_set, beta_set))
psi_est_all3 = np.array(grad_f(Z3, map_matrix_set, feature_set, threshold_set, beta_set))
# psi_est_all = np.array(grad_f(Z, map_matrix_set, feature_set, threshold_set, beta_set))
psi_est_all = np.concatenate([psi_est_all1, psi_est_all2, psi_est_all3], axis=0)
grad_train = np.mean(psi_est_all ** 2, axis=0)
columns = ['clinic', 'sex', 'prot', 'fat', 'carb', 'fib', 'ash', 'as_ln', 'mn_ln', 'pb_ln']
columns = ['Long pos', 'Prismatic coeff', 'Length-displacement ratio', 'Beam-draught ratio',
           'Length-beam ratio', 'Froude number']
columns = ['Cement', 'Blast Furnace Slag', 'Fly Ash', 'Water', 'Superplasticizer',
           'Coarse Aggregate', 'Fine Aggregate', 'Age']
# dim = 25
# for j in range(dim):
#     columns.append("x" + str(j + 1))

df_total = pd.DataFrame(grad_train, columns=columns)
df_total.to_csv("/Users/irisdeng/Desktop/fdt_vi_c1000.csv", index=False)
psi_est = np.median(grad_train, axis=0)


# collect feature importances from ras random forest
feature_imp = np.zeros((rf_raw.n_estimators, Z.shape[1]))
for j, model in enumerate(rf_raw.estimators_):
    feature_imp[j, :] = model.feature_importances_

df_total = pd.DataFrame(feature_imp, columns=columns)
df_total.to_csv("/Users/irisdeng/Desktop/rf_vi.csv", index=False)


# categorical interacts with continuous
x1 = np.copy(Z_test)
x1[:, 0] = 1
x1[:, 1] = 1
y1_all = np.array(f(x1, map_matrix_set, feature_set, threshold_set, beta_set))
y11 = np.mean(y1_all, axis=1).reshape(-1, 1)

x1 = np.copy(Z_test)
x1[:, 0] = 1
x1[:, 1] = 2
y1_all = np.array(f(x1, map_matrix_set, feature_set, threshold_set, beta_set))
y10 = np.mean(y1_all, axis=1).reshape(-1, 1)

x1 = np.copy(Z_test)
x1[:, 0] = 2
x1[:, 1] = 1
y1_all = np.array(f(x1, map_matrix_set, feature_set, threshold_set, beta_set))
y21 = np.mean(y1_all, axis=1).reshape(-1, 1)

x1 = np.copy(Z_test)
x1[:, 0] = 2
x1[:, 1] = 2
y1_all = np.array(f(x1, map_matrix_set, feature_set, threshold_set, beta_set))
y20 = np.mean(y1_all, axis=1).reshape(-1, 1)

y_all = np.concatenate([y11, y10, y21, y20], axis=1)
fig, ax = plot_slices_5(Z_test, y_all, quantile=.5, figsize=(10, 12))
fig.savefig("/Users/irisdeng/Downloads/bangladesh/cat_cont.png")




# adult dataset
features = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
            "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
            "hours_per_week", "native_country", "label"]

original_train = pd.read_csv("/Users/wdeng/Desktop/FDT/simu_real/adult/adult.data", names=features,
                             sep=r'\s*, \s*', engine="python", na_values="?")
original_test = pd.read_csv("/Users/wdeng/Desktop/FDT/simu_real/adult/adult.test", names=features,
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

def data_transform_integer(df):
    """Normalize features."""
    cat_df = df.select_dtypes(include=['object']).copy()
    for col_name in list(cat_df.columns):
        cat_df[col_name] = cat_df[col_name].astype('category')
        cat_df[col_name] = cat_df[col_name].cat.codes
    cont_features = list(set(df.columns) - set(cat_df.columns))
    cont_df = df[cont_features].copy()
    scaler = preprocessing.StandardScaler()
    cont_df = pd.DataFrame(scaler.fit_transform(cont_df), columns = cont_features)
    data = pd.concat([cont_df, cat_df.reset_index(drop=True)], axis=1)
    return data

# feature_mat = data_transform_integer(original)


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



# heart disease data
features = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
            "exang", "oldpeak", "slope", "ca", "thal", "target"]

original = pd.read_csv("/Users/wdeng/Desktop/FDT/simu_real/heart/processed.cleveland.data", names=features,
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


# Myocardial infarction complications Data Set
original = pd.read_csv("/Users/wdeng/Desktop/FDT/simu_real/mi/MI.data", engine="python",
                       na_values="?", index_col=0, header=None)
features = ["sex", "ritm_ecg_p_01", "age", "s_ad_orit", "d_ad_orit",
            "ant_im", "ibs_post", "k_blood", "na_blood", "l_blood"]

feature_mat = original[[2, 49, 1, 36, 37, 44, 6, 83, 85, 89]].copy()
feature_mat = feature_mat.dropna()
feature_mat.rename({2: "sex", 49: "ritm_ecg_p_01", 1: "age", 36: "s_ad_orit", 37: "d_ad_orit",
                    44: "ant_im", 6: "ibs_post", 83: "k_blood", 85: "na_blood", 89: "l_blood"},
                   axis = "columns", inplace = True)
corrMat = feature_mat.corr().round(2)
ax = sn.heatmap(corrMat, annot=True)
ax.add_patch(Rectangle((0,0), 5, 5, fill=False, edgecolor='k', lw=3, clip_on=False))
plt.tight_layout()
plt.show()

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
feature_mat.to_csv("/Users/wdeng/Desktop/FDT/simu_real/mi/feature_processed.csv",
                   columns=features, index=False)

# Myocardial infarction complications Data Set with imputed values
feature_mat = pd.read_csv("/Users/wdeng/Desktop/FDT/simu_real/mi/feature_processed.csv")
feature_mat = feature_mat.iloc[:, :20]
corrMat = feature_mat.corr().round(2)
ax = sn.heatmap(corrMat, annot=True, cmap='rocket_r', annot_kws={'size':7})
ax.add_patch(Rectangle((0,0), 5, 5, fill=False, edgecolor='k', lw=3, clip_on=False))
plt.tight_layout()
plt.show()


