# Now, use 10-fold stratified cross-validation again but apply 
# preprocessing to each fold (not to the entire data) before 
# building the tree.


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
import numpy as np
import pandas as pd


df = pd.read_csv('diabetes_risk.csv')

df = df.drop(columns=['patient_id'])

X = df.drop(columns=['diabetes_risk'])
y = df['diabetes_risk']


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
    ('tree', DecisionTreeClassifier(random_state=0))
])

skf = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=0
)

cv_scores = cross_validate(
    Model,
    X,
    y, 
    scoring=[
        'accuracy',
        'precision_macro',
        'recall_macro',
        'f1_macro'
    ],
    cv=skf
)

metrics = [
    'test_accuracy',
    'test_precision_macro',
    'test_recall_macro',
    'test_f1_macro'
]

for metric in metrics:
    print(metric)
    print(cv_scores[metric])
    print("Mean:", cv_scores[metric].mean())
    print("Standard deviation:", cv_scores[metric].std())
    print()