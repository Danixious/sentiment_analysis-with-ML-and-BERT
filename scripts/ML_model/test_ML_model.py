import pandas as pd
import joblib
model = joblib.load("D:/sentiment_analysis/data/temp_data/sentiment_model.pkl")
vectorizer = joblib.load("D:/sentiment_analysis/data/temp_data/bow_vectorizer.pkl")

test_cases = ["the movie was nice and enjoyable",
              "i really enjoyed it",
              "i have never seen a worse movie in my life",
              "sure",
              "this movie was lit",
              "the movie was like a burning sensation",
              "banging movie ngl",
              "shit movie",
              "haram",
              "fire",
              "meh"
              ]

X_test = vectorizer.transform(test_cases)
prediction = model.predict(X_test)
label_map = {0:"Negative",1:"Neutral",2:"Positive"}
for review, pred in zip(test_cases, prediction):
    print(f"Review: {review} -> Sentiment: {label_map[pred]}")
