## Jordan Dubchak, Jan 2018
##
## Clean the downloaded files, remove NA values and saved cleaned data files to data/clean_data 

## necessary library imports 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

## read in each raw data file and replace ? characters with NaN values 
clev = pd.read_csv("data/raw_data/cleveland.csv", index_col=0, na_values="?")
hung = pd.read_csv("data/raw_data/hungary.csv", index_col=0, na_values="?")
va = pd.read_csv("data/raw_data/longbeachva.csv", index_col=0, na_values="?")
switz = pd.read_csv("data/raw_data/switzerland.csv", index_col=0, na_values="?")

## include a col in each data frame for the location the data was collected  
clev["Location"] = ["Cleveland"]*len(clev)
hung["Location"] = ["Hungary"]*len(hung)
va["Location"] = ["Long Beach VA"]*len(va)
switz["Location"] = ["Switzerland"]*len(switz)

## combine all four data frames to one and reset the index to remove the extra col it creates by default 
combined = pd.concat([clev, hung, va, switz])
combined = combined.reset_index(drop=True)

## EDA: visualize NaN values by feature and save 
fig1 = sns.heatmap(combined.isnull(), yticklabels=False, cmap="plasma")
fig1 = plt.tight_layout()
fig1 = sns.plt.title("Presence of NaN Values")
figure = fig1.get_figure()
figure.savefig("results/nanvals_EDA.png", dpi=400)

## how many NaN values are we dealing with?
combined.isnull().sum()

## remove slope, ca, and thal (all contain too many missing values)
combined = combined.drop(["ca", "thal", "slope"], axis=1)

## what are the feature types?
combined.info()

## table of null values after removal of 3 features 
null_vals = pd.DataFrame(combined.isnull().sum(axis=0))
null_vals.columns = ["null_val_count"]
null_vals
null_vals["percent_null_vals"] = round((null_vals["null_val_count"]/len(combined))*100,2)
null_vals.to_csv("data/clean_data/null_val_table.csv")

## drop missing values and reset index 
combined = combined.dropna(axis=0)
combined = combined.reset_index(drop=True)

## change feature types
cols = ["sex", "age", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang"]

for col in cols:
    combined[col] = combined[col].astype(int)

combined["oldpeak"] = combined["oldpeak"].astype(float)

## create distributions of each feature 
plot1 = sns.set(style="white")
f, axes = plt.subplots(3,2, figsize=(12,10), sharey=False, sharex=False)
plot1 =sns.distplot(combined.chol, ax=axes[0,0])
plot1 = sns.distplot(combined.trestbps, ax=axes[0,1])
plot1 = sns.distplot(combined.thalach, ax=axes[1,0])
plot1 = sns.distplot(combined.oldpeak, ax=axes[1,1])
plot1 = sns.distplot(combined.age, ax=axes[2,0])
plot1 = sns.distplot(combined.cp, ax=axes[2,1])
figure = plot1.get_figure()
figure.savefig("results/cont_eda.png", dpi=400)

## create count plots of discrete features 
plot2 = sns.set(style="white")
f, axes = plt.subplots(2,2, figsize=(10,10), sharey=False, sharex=False)
plot2 = sns.countplot(combined.exang, ax=axes[0,0])
plot2 = sns.countplot(combined.restecg, ax=axes[0,1])
plot2 = sns.countplot(combined.fbs, ax=axes[1,0])
plot2 = sns.countplot(combined.sex, ax=axes[1,1])
figure = plot2.get_figure()
figure.savefig("results/discrete_eda.png", dpi=400) 

## convert restecg to dummy variables 
rest_ecg = pd.get_dummies(combined.restecg)
rest_ecg.columns = ["restecg_normal", "restecg_anbormal_stt", "restecg_hypertrophy"]
rest_ecg.head()
combined["restecg_normal"] = rest_ecg.restecg_normal
combined["restecg_anbormal_stt"] = rest_ecg.restecg_anbormal_stt
combined = combined.drop("restecg", axis=1)

## convert cd to dummy variables 
cp = pd.get_dummies(combined.cp)
cp.columns= ["cp1_typical_angina", "cp2_atypical_angina", "cp3_non-anginal_pain", "cp4_asymptomatic"]
combined["cp1_typical_angina"] = cp["cp1_typical_angina"]
combined["cp2_atypical_angina"] = cp["cp2_atypical_angina"]
combined["cp3_non-anginal_pain"] = cp["cp3_non-anginal_pain"]
combined = combined.drop("cp", axis=1)

## save to csv 
combined.to_csv("data/clean_data/heart_disease_cases.csv")