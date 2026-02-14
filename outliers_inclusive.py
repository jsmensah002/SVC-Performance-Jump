import pandas as pd
df = pd.read_csv('Bank_Churn.csv')
print(df)

print(df.isna().sum())

category_column = df[['Surname','Geography','Gender']]
numerical_column = df[['CreditScore','Age','Tenure','Balance','NumOfProducts']]
binary_column = df[['IsActiveMember','HasCrCard']]

category_dummies = pd.get_dummies(category_column,dtype=float)
print(category_dummies)

surname_dummies = pd.get_dummies(df['Surname'],dtype=float)
geography_dummies = pd.get_dummies(df['Geography'],dtype=float)
gender_dummies = pd.get_dummies(df['Gender'],dtype=float)

x = pd.concat([category_dummies,
               numerical_column,
               binary_column],axis='columns')
y =df['Exited']

print(x)
print(y)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(class_weight='balanced',n_estimators=100,random_state=42)
rf.fit(x,y)

import sklearn.linear_model as lm
logreg = lm.LogisticRegression(class_weight='balanced',max_iter=1000)
logreg.fit(x,y)

from sklearn.svm import SVC
svm = SVC(class_weight='balanced')
svm.fit(x,y)

from sklearn.model_selection import train_test_split as tt
x_train,x_test,y_train,y_test = tt(x,y,test_size=0.2,random_state=42)
logreg.fit(x_train,y_train)
svm.fit(x_train,y_train)
rf.fit(x_train,y_train)

print(logreg.score(x,y))
print(logreg.score(x_train,y_train))
print(logreg.score(x_test,y_test))

print(svm.score(x,y))
print(svm.score(x_train,y_train))
print(svm.score(x_test,y_test))

print(rf.score(x,y))
print(rf.score(x_train,y_train))
print(rf.score(x_test,y_test))