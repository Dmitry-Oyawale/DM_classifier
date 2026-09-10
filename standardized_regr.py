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

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])




# Dummy regressor
from sklearn.dummy import DummyRegressor

dummy_regr = DummyRegressor(strategy="mean")
dummy_regr.fit(X_train, y_train)

print("\nDummy Regressor")
print("Predicted BMI:", dummy_regr.constant_[0])

# Linear Regression

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

# Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor

dtr = DecisionTreeRegressor(random_state=0)
dtr.fit(X_train, y_train)

print("dtr")
print("Decision Tree depth:", dtr.get_depth())
print("Decision Tree leaves:", dtr.get_n_leaves())

dtr_importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': dtr.feature_importances_
}).sort_values(by='importance', ascending=False)

top_features = dtr_importance_df.head(10)

plt.figure(figsize=(10, 6))
plt.barh(top_features['feature'], top_features['importance'])
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Decision Tree Feature Importances')
plt.gca().invert_yaxis()
plt.show()

# Random Forest Regressor

from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(random_state=0)

rfr.fit(X_train, y_train)

rf_importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': rfr.feature_importances_
}).sort_values(by='importance', ascending=False)

print("\nrfr")

print(rf_importance_df.head(10))

top_features = rf_importance_df.head(10)

plt.figure(figsize=(10, 6))
plt.barh(top_features['feature'], top_features['importance'])
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Random Forest Feature Importances')
plt.gca().invert_yaxis()
plt.show()