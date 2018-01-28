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
train_dat = pd.read_csv("data/clean_data/training_data.csv", index_col=0)

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

## instructor code 
mean_squared_err = lambda y, yhat: np.mean((y-yhat)**2)

## to store results 
results = pd.DataFrame(columns = ["feature_selection", "model", "error_test", "train_err", "val_err", "test_err"])

## using Pipeline, SelectFromModel, LinearSVC
mod1 = Pipeline([
    ("feature_selection", SelectFromModel(LogisticRegression(penalty="l1", random_state=1234, solver="saga", tol=0.01))), 
    ("classification", LinearSVC(penalty="l1", dual=False))
])
mod1.fit(X_train, y_train)

train_mod1 = mod1.score(X_train, y_train)
val_mod1 = mod1.score(X_val, y_val)
test_mod1 = mod1.score(X_test, y_test)

results.loc[0] = ["Select From Model - LogisticRegression", "LinearSVC", "Logistic Regression", train_mod1, val_mod1, test_mod1]

## just linearSVC
cvals={"C": np.logspace(-3,3,20)}
svc = LinearSVC(random_state=1234, penalty="l1", multi_class="crammer_singer")
mod2 = GridSearchCV(svc, cvals)
mod2.fit(X,y)
best_c = mod2.best_params_["C"]

## now take the fine tuned C value and fit the model on all of X and y
mod2svc = LinearSVC(random_state=1234, penalty="l1", multi_class="crammer_singer", C =best_c)
mod2svc.fit(X, y)
train_mod2svc = mod2svc.score(X_train,y_train)
val_mod2svc = mod2svc.score(X_val, y_val)
test_mod2 = mod2svc.score(X_test, y_test)

results.loc[1] = ["LinearSVC", "LinearSVC", "LinearSVC", train_mod2svc, val_mod2svc, test_mod2]

## knn
estimator = LinearSVC()
selector = RFE(estimator)
selector = selector.fit(X,y)
selector.ranking_
## get the relevant cols 
relevant_cols = X.columns[np.where(selector.ranking_ == 1)]
X = X[relevant_cols]

## now grid search over the relevant cols for the best number of neighbours
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42)
X_test = X_test[relevant_cols]

params = {"n_neighbors": [val for val in range(1,40)]}
knn = KNeighborsClassifier()
clf = GridSearchCV(knn, params)
clf.fit(X,y)
best_num_neighs = clf.best_params_["n_neighbors"]

knn = KNeighborsClassifier(n_neighbors=best_num_neighs)
knn.fit(X, y)
train_knn = knn.score(X_train, y_train)
val_knn = knn.score(X_val, y_val)
test_knn = knn.score(X_test, y_test)

results.loc[2] = ["RFE LinearSVC", "KNN", "KNN", train_knn, val_knn, test_knn]

results.to_csv("data/clean_data/results.csv")


