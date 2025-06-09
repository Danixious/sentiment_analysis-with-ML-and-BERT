import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,classification_report
import joblib
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def train_test_data():

    data = pd.read_csv("D:/sentiment_analysis/data/cleaned_data_ML.csv")
    X_bow = joblib.load("D:/sentiment_analysis/data/temp_data/X_bow.pkl")
    print(type(X_bow))
    X = X_bow
    y = data["Sentiment_label"]

    mask = ~y.isna().to_numpy()
    X = X[mask]
    y = y[mask].reset_index(drop=True)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=42)
    return X_train,X_test,y_train,y_test

vectorizer = joblib.load("D:/sentiment_analysis/data/temp_data/bow_vectorizer.pkl")
X_bow = joblib.load("D:/sentiment_analysis/data/temp_data/X_bow.pkl")

if __name__ == "__main__":

#implementation of Naive Bayes
   X_train,X_test,y_train,y_test = train_test_data()
   model   = MultinomialNB()
   model  = model.fit(X_train,y_train)
   y_pred = model.predict(X_test)
   print("accuracy score:  ",accuracy_score(y_test,y_pred))
   print("Classification Report:\n",classification_report(y_test,y_pred))

   joblib.dump(model, "D:/sentiment_analysis/data/temp_data/sentiment_model.pkl")

   cm = confusion_matrix(y_test,y_pred)
   disp = ConfusionMatrixDisplay(confusion_matrix = cm,display_labels=["Negative","Neutral","Positive"])
   disp.plot()
   plt.title("confusion Matrix")
   plt.show()

