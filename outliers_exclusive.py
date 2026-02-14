#Importing file using pandas
import pandas as pd
df = pd.read_csv('Bank_Churn.csv')
print(df)

import numpy as np

numerical_column = df[['CreditScore','Age','Tenure','Balance','NumOfProducts']]
binary_column = df[['IsActiveMember','HasCrCard']]

for col in numerical_column:
    lower = df[col].quantile(0.01)
    upper = df[col].quantile(0.99)
    df[col] = df[col].where((df[col]>=lower) & (df[col]<=upper),np.nan)

print(df)

categorical = df[['Surname','Geography','Gender']]

categorical_dummies = pd.get_dummies(categorical,dtype=float)

x = pd.concat([categorical_dummies,
               numerical_column,
               binary_column],axis='columns')

y = df['Exited']

print(x)
print(y)

print(df.isna().sum())

df['CreditScore'] = df['CreditScore'].fillna(df['CreditScore'].median())
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Balance'] = df['Balance'].fillna(df['Balance'].median())
df['NumOfProducts'] = df['NumOfProducts'].fillna(df['NumOfProducts'].median())

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print("\n=== CLASSIFICATION MODELS ===\n")

# Logistic Regression (scaled)
log_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=1000))
])
log_grid = GridSearchCV(
    log_pipe,
    {
        "logreg__C": [0.01, 0.1, 1, 10],
        "logreg__solver": ["lbfgs"],
        "logreg__class_weight": [None,'balanced']
    },
    cv=5,
    scoring="accuracy",
    n_jobs=1
)
log_grid.fit(x_train, y_train)
print("Logistic Regression best params:", log_grid.best_params_)
best_logreg = log_grid.best_estimator_
print('Train R2:',best_logreg.score(x_train,y_train))
print('Test R2:',best_logreg.score(x_test,y_test))

# SVC (scaled)
svc_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC())
])
svc_grid = GridSearchCV(
    svc_pipe,
    {
        "svc__kernel": ["linear", "rbf"],
        "svc__C": [0.1, 1, 10],
        "svc__gamma": ["scale"]
    },
    cv=5,
    scoring="accuracy",
    n_jobs=1
)
svc_grid.fit(x_train, y_train)
print("SVC best params:", svc_grid.best_params_)

best_svc = svc_grid.best_estimator_
print('Train R2:',best_svc.score(x_train,y_train))
print('Test R2:',best_logreg.score(x_test,y_test))

# Random Forest Classifier
rfc_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    {
        "n_estimators": [100, 300],
        "max_depth": [None, 10],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 3],
        "max_features": ["sqrt"]
    },
    cv=5,
    scoring="accuracy",
    n_jobs=1
)
rfc_grid.fit(x_train, y_train)
print("Random Forest Classifier best params:", rfc_grid.best_params_)

best_rfc = rfc_grid.best_estimator_
print('Train R2:',best_rfc.score(x_train,y_train))
print('Test R2:',best_logreg.score(x_test,y_test))
