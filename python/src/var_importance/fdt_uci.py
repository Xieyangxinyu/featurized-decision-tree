import pandas as pd
import numpy as np
import csv

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
