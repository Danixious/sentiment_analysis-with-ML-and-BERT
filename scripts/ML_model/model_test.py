import pandas as pd
from sklearn.linear_model import LogisticRegression
from model import train_test_data
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier




X_train,X_test,y_train,y_test = train_test_data()
model   = LogisticRegression()
model  = model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print("accuracy score:  ",accuracy_score(y_test,y_pred))

model = RandomForestClassifier()
model  = model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print("Accuracy Score:  ",accuracy_score(y_test,y_pred))