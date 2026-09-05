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

cat_imputer = SimpleImputer(strategy='constant', fill_value='Unknown')
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
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


DumC = DummyClassifier(strategy='most_frequent')

DumC.fit(X_train, y_train)
# print(DumC.predict(X_test))
# print(DumC.score(X_test, y_test))

# default
dt1 = DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_split=2)

dt1.fit(X_train, y_train)
# print(dt1.predict(X_test))
# print(dt1.score(X_test, y_test))

print("dt1")
print("Decision Tree depth:", dt1.get_depth())
print("Decision Tree leaves:", dt1.get_n_leaves())

importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': dt1.feature_importances_
}).sort_values(by='importance', ascending=False)

print(importance_df.head(10))

dt_importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': dt1.feature_importances_
}).sort_values(by='importance', ascending=False)

top_features = dt_importance_df.head(10)

plt.figure(figsize=(10, 6))
plt.barh(top_features['feature'], top_features['importance'])
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Decision Tree Feature Importances')
plt.gca().invert_yaxis()
plt.show()


from sklearn.ensemble import RandomForestClassifier

# default
rf1 = RandomForestClassifier(n_estimators=100, max_depth=None)

rf1.fit(X_train, y_train)
# print(rf1.predict(X_test))
# print(rf1.score(X_test, y_test))

rf_importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': rf1.feature_importances_
}).sort_values(by='importance', ascending=False)

print("\nrf1")

print(rf_importance_df.head(10))

top_features = rf_importance_df.head(10)

plt.figure(figsize=(10, 6))
plt.barh(top_features['feature'], top_features['importance'])
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Random Forest Feature Importances')
plt.gca().invert_yaxis()
plt.show()