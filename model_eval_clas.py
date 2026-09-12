import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import sklearn
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

from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier



from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

DT = DecisionTreeClassifier(random_state=0)

DT.fit(X, y)
print(DT.get_depth())

Model = DT # I"m doing this assignment so that I can reuse the code below with other models

# Testing Model on the training set:
print("Accuracy: ", Model.score(X, y))

y_pred = Model.predict(X)

CM = confusion_matrix(y, y_pred)
print(CM)

# Visualizing the confusion matrix

CMviz = ConfusionMatrixDisplay(confusion_matrix=CM, display_labels=['Low', 'Moderate', 'High'])
CMviz.plot(cmap='Greens')

plt.show()

# Performance metrices
print(classification_report(y, y_pred))


from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate

Model = DT

# Using Stratified Cross-Validation
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

# Using cross_validate to obtain all the desired evaluation metrics
# Note that using cross_val_score yields only the default evaluation metric (accuracy)

cv_scores = cross_validate(Model, X, y, scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'], cv=skf)

print("StratifiedK-fold scores of each fold: ", cv_scores)

#include results (mean and standard deviation) in report
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

# If you need to generate predictions on each test fold
y_pred = cross_val_predict(Model, X, y, cv=skf)
print(classification_report(y, y_pred))

CM = confusion_matrix(y, y_pred)
print(CM)

# Visualizing the confusion matrix

CMviz = ConfusionMatrixDisplay(confusion_matrix=CM, display_labels=['Low', 'Moderate', 'High'])
CMviz.plot(cmap='Greens')

plt.show()