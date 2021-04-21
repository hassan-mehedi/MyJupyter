import numpy as np
import pandas as pd 
from operator import itemgetter

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import warnings
import pickle

warnings.filterwarnings("ignore")

df = pd.read_csv("crop_prediction_model_one.csv")

X = df.drop("label", axis = 1)
y = df["label"]

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

grid = {"n_estimators": [10, 100, 200, 500, 1000, 1200],
        "max_depth": [None, 5, 10, 20, 30],
        "max_features": ["auto", "sqrt"],
        "min_samples_split": [2, 4, 6],
        "min_samples_leaf": [1, 2, 4]}

clf = RandomForestClassifier(n_jobs=1)

# Setup RandomizedSearchCV
rs_clf = RandomizedSearchCV(estimator=clf,
                            param_distributions=grid, 
                            n_iter=10, # number of models to try
                            cv=5,
                            verbose=2)

# Fit the RandomizedSearchCV version of clf
rs_clf.fit(X_train, y_train)
params = rs_clf.best_params_

n_estimators, min_samples_split, min_samples_leaf, max_features, max_depth = itemgetter(
     "n_estimators",
     "min_samples_split",
     "min_samples_leaf",
     "max_features",
     "max_depth"
 )(params)

# Initializing our model with the best parameters
# Parameters were found out by using RandomizedSearchCV
rfc = RandomForestClassifier(n_estimators=n_estimators, min_samples_split=min_samples_split,
                              min_samples_leaf=min_samples_leaf, max_features=max_features,
                              max_depth=max_depth)
rfc.fit(X_train, y_train)
pickle.dump(rfc, open('model.pkl', 'wb'))