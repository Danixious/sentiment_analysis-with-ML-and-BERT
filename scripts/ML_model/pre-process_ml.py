import pandas as pd
import numpy as np
import string
import nltk

from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer

# Load the dataset (you used this path)
data = pd.read_csv("D:/sentiment_analysis/data/labeled_data.csv", encoding='ISO-8859-1', on_bad_lines='skip')

print("\n Columns loaded from labeled_data.csv:")
print(data.columns.tolist())

# Rename or remove unused columns
if 'Review' in data.columns:
    data.drop(columns=['Review'], inplace=True)

# Check for sentiment distribution
print("Sentiment Label distribution:")
print(data['Sentiment'].value_counts(dropna=False))

# Drop missing Text entries and ensure type
data = data.dropna(subset=["Text"])
data["Text"] = data["Text"].astype(str)

# Lowercase
def clean_text_columns(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.lower()
    return df

# Punctuation removal
def remove_punctuation(text):
    if isinstance(text, str):
        return text.translate(str.maketrans('', '', string.punctuation))
    return text

def clean_dataframe_punctuation(df):
    str_cols = df.select_dtypes(include=['object', 'string']).columns
    for col in str_cols:
        df[col] = df[col].apply(remove_punctuation)
    return df

# Apply cleaning
data = clean_text_columns(data)
data = clean_dataframe_punctuation(data)
print(data.head())

# Download NLTK resources
nltk.download('wordnet')
nltk.download('omw-1.4')

# Lemmatization
w_tokenizer = WhitespaceTokenizer()
lemmatizer = WordNetLemmatizer()

def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]

data['text_lemmatize'] = data['Text'].apply(lemmatize_text)

print("\nSample after lemmatization:")
print(data['text_lemmatize'].head(10))
print("Any empty rows?", sum(data['text_lemmatize'].apply(lambda x: len(x) == 0)))

# Final output
print(data.head())

# Save cleaned data
data.to_csv("D:/sentiment_analysis/data/cleaned_data_ML.csv", index=False)
print(" cleaned data has been saved!")
