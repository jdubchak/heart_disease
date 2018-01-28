## Jordan Dubchak, 2018

import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline 
from sklearn.feature_selection import RFE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

## read in data
train_dat = pd.read_csv("data/clean_data/training_data.csv")

## split labels and features 
y = train_dat["label"]
y = y.tolist() ## was an unknown class label, now is a multiclass input 
X = train_dat.drop("label", axis=1)
X = X.drop("Location", axis=1)

## read in test data
test_dat = pd.read_csv("data/clean_data/test_data.csv", index_col=0)

y_test = test_dat["label"]
y_test = y_test.tolist() ## was an unknown class label, now is a multiclass input 
X_test = test_dat.drop("label", axis=1)
X_test = X_test.drop("Location", axis=1)

## split into train and validation sets 
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=1234)

## to store results 
results2 = pd.DataFrame(columns = ["model", "train_err", "val_err", "test_err"])

## LinearSVC, no model selection 
svc_mod = LinearSVC(penalty="l1", multi_class="crammer_singer", dual=False)
params = {"C": np.logspace(-3, 3, 20)}
clf = GridSearchCV(svc_mod, params)
clf.fit(X,y)

## fit with best C
svc_mod2 = LinearSVC(penalty="l1", multi_class="crammer_singer", dual=False, C=clf.best_params_["C"])
clf.fit(X,y)
train_svc = clf.score(X_train, y_train)
val_svc = clf.score(X_val, y_val)
test_svc = clf.score(X_test, y_test)

results2.loc[0] = ["LinearSVC", train_svc, val_svc, test_svc]

## knn, find best number of neighbours 
params = {"n_neighbors": [val for val in range(1,40)]}
knn = KNeighborsClassifier()
clf = GridSearchCV(knn, params)
clf.fit(X,y)

## fit with best number of neighbours 
knn = KNeighborsClassifier(n_neighbors=clf.best_params_["n_neighbors"])
knn.fit(X, y)

train_knn = knn.score(X_train, y_train)
val_knn = knn.score(X_val, y_val)
test_knn = knn.score(X_test, y_test)

results2.loc[1] = ["KNN", train_knn, val_knn, test_knn]

## save to csv
results2.to_csv("heart_disease-/data/clean_data/results2.csv")