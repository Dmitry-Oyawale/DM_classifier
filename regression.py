import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import sklearn
from sklearn.impute import SimpleImputer

df = pd.read_csv('diabetes_risk.csv')

df = df.drop(columns=['patient_id'])

X = df.drop(columns=['diabetes_risk', 'bmi'])
y = df['bmi']

categorical_cols = X.select_dtypes(include='object').columns
numeric_cols = X.select_dtypes(include=np.number).columns

cat_imputer = SimpleImputer(strategy='most_frequent')
X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])

num_imputer = SimpleImputer(strategy='mean')
X[numeric_cols] = num_imputer.fit_transform(X[numeric_cols])

X = pd.get_dummies(
    X,
    columns=categorical_cols,
    drop_first=False,
    dtype=int
)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)





# Dummy regressor
from sklearn.dummy import DummyRegressor

dummy_regr = DummyRegressor(strategy="mean")
dummy_regr.fit(X_train, y_train)

print("\nDummy Regressor")
print("Predicted BMI:", dummy_regr.constant_[0])

from sklearn.linear_model import LinearRegression
reg = LinearRegression()

reg.fit(X_train,y_train)

print("coefficients")
print(reg.coef_)

print("intercept")
print(reg.intercept_)

print("\nFormula:")

formula = f"BMI = {reg.intercept_:.4f}"

for feature, coef in zip(X.columns, reg.coef_):
    if coef >= 0:
        formula += f" + {coef:.4f}*{feature}"
    else:
        formula += f" - {abs(coef):.4f}*{feature}"

print(formula)