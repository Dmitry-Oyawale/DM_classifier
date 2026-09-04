import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.impute import SimpleImputer

df = pd.read_csv('diabetes_risk.csv')

df = df.drop(columns=['patient_id'])

X = df.drop(columns=['diabetes_risk'])
y = df['diabetes_risk']

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