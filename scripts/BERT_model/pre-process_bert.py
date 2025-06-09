import pandas as pd
import numpy as np

data = pd.read_csv("D:/sentiment_analysis/data/labeled_data.csv",encoding = 'ISO-8859-1',on_bad_lines='skip')
print("\n Columns loaded from labeled_data.csv:")
print(data.columns.tolist())

#conversion of dtype
data = data.dropna(subset = ["Text"])
data["Text"] = data["Text"].astype(str)


#lowercasing
def clean_text_coloumns(data):
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = data[col].astype(str).str.lower()
    return data

print(data.head())

data.to_csv("D:/sentiment_analysis/data/cleaned_data_bert.csv")

print("cleaned_data has been saved!")






