# 2.2. Now, use 10-fold cross-validation again but apply 
# preprocessing to each fold (not to the entire data) before 
# building the tree. Same as you did in Part I above.


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import StratifiedKFold, KFold, cross_validate, cross_val_predict
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt



df = pd.read_csv('diabetes_risk.csv')

df = df.drop(columns=['patient_id'])

df = df.dropna(subset=['bmi'])

X = df.drop(columns=['bmi'])
y = df['bmi']

categorical_cols = X.select_dtypes(include='object').columns
numeric_cols = X.select_dtypes(include=np.number).columns

preprocessor = ColumnTransformer([
    (
        'cat',
        Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]),
        categorical_cols
    ),
    (
        'num',
        Pipeline([
            ('imputer', SimpleImputer(strategy='mean'))
        ]),
        numeric_cols
    )
])

Model = Pipeline([
    ('preprocessor', preprocessor),
    ('tree', DecisionTreeRegressor(random_state=0))
])

kcv = KFold(
    n_splits=10,
    shuffle=True,
    random_state=0
)

cv_scores = cross_validate(
    Model,
    X,
    y, 
    scoring=[
        'neg_mean_squared_error', 
        'neg_root_mean_squared_error', 
        'neg_mean_absolute_error', 
        'r2'
    ],
    cv=kcv
)

metrics = [
    'test_neg_mean_squared_error', 
    'test_neg_root_mean_squared_error', 
    'test_neg_mean_absolute_error', 
    'test_r2'
]

for metric in metrics:
    print(metric)
    print(cv_scores[metric])
    print("Mean:", cv_scores[metric].mean())
    print("Standard deviation:", cv_scores[metric].std())
    print()

# If you need to generate predictions on each test fold
y_pred = cross_val_predict(Model, X, y, cv=kcv)
