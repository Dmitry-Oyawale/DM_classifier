import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import sklearn
from sklearn.impute import SimpleImputer

df = pd.read_csv('diabetes_risk.csv')

df = df.drop(columns=['patient_id'])

X = df.drop(columns=['bmi'])
y = df['bmi']

categorical_cols = X.select_dtypes(include='object').columns
numeric_cols = X.select_dtypes(include=np.number).columns

cat_imputer = SimpleImputer(strategy='most_frequent')
X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])

num_imputer = SimpleImputer(strategy='mean')
X[numeric_cols] = num_imputer.fit_transform(X[numeric_cols])
y = num_imputer.fit_transform(y.to_frame()).ravel()

X = pd.get_dummies(
    X,
    columns=categorical_cols,
    drop_first=False,
    dtype=int
)

from sklearn.tree import DecisionTreeRegressor

DT = DecisionTreeRegressor(random_state=0)

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, root_mean_squared_error, explained_variance_score

# EVALUATING REGRESSION PERFORMANCE ON THE TRAINING SET

Model = DT

# Testing Model on the training set:

Model.fit(X, y)

# Generate predictions

y_pred = Model.predict(X)

# Evaluate performance

mse = mean_squared_error(y, y_pred)

rmse = root_mean_squared_error(y, y_pred)

mae = mean_absolute_error(y, y_pred)

r2 = r2_score(y, y_pred)

ev_score = explained_variance_score(y, y_pred)

print(f"Mean Squared Error:  {mse:.2f}")

print(f"Root Mean Square Error: {rmse:.2f}")

print(f"Mean Absolute Error: {mae:.2f}")

print(f"R² Score:            {r2:.2f}")

print(f"Explained Variance:  {ev_score:.2f}")

# EVALUATING PREDICTIVE PERFORMANCE USING K-FOLD CROSS-VALIDATION

# Before assign to Model the particular model you constructed above 
# that you want to evaluate. For example,
# Model = dummy_majority
# Model = DT
# and so on.

# Model = dummy_majority

Model = DT

from sklearn.model_selection import KFold, cross_validate, cross_val_predict

# Using plain Cross-Validation
# Note Stratified Cross-laidation is not implemented for regression in scikit-learn
# Using cross-validate to obtain all the desired evaluation metrics
# Note: using cross_val_score instead only produces the default evaluation metric (R^2)

kcv = KFold(n_splits=10, shuffle=True, random_state=0)

cv_scores = cross_validate(Model, X, y, scoring=['neg_mean_squared_error', 'neg_root_mean_squared_error', 'neg_mean_absolute_error', 'r2'], cv=kcv)

print("\n ***** Using cross-validate *****\n")
print("K-fold Scores of each fold: ", cv_scores)

# If you need to predict the test fold target values

y_pred = cross_val_predict(Model, X, y, cv=kcv)