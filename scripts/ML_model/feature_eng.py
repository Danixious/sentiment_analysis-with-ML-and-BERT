import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import joblib
import ast

# Load cleaned data
data = pd.read_csv("D:/sentiment_analysis/data/cleaned_data_ML.csv")

# Join token lists into strings
def safe_join(x):
    try:
        if isinstance(x, str):
            tokens = ast.literal_eval(x)
            if isinstance(tokens, list):
                return ' '.join(tokens)
    except Exception as e:
        print(f"Error parsing: {e}")
        return ''
    return ''

data['bow_text'] = data['text_lemmatize'].apply(safe_join)

# Filter out empty strings
print("Empty rows before filtering:", sum(data['bow_text'].str.strip() == ''))
data = data[data['bow_text'].str.strip() != '']
print("Empty rows after filtering:", sum(data['bow_text'].str.strip() == ''))

# Sample check
example = data['text_lemmatize'].iloc[0]
tokens = ast.literal_eval(example) if isinstance(example, str) else []
print("Tokens:", tokens)

# Create BoW vectorizer
vectorizer = CountVectorizer(stop_words='english')
X_bow = vectorizer.fit_transform(data['bow_text'])

# Output details
print("Vocabulary sample:", list(vectorizer.vocabulary_.items())[:10])
print("\n BoW Matrix Shape:", X_bow.shape)
print("\nSample Encoded Row:", X_bow.toarray()[0])

# Save vectorizer and transformed matrix
joblib.dump(X_bow, "D:/sentiment_analysis/data/temp_data/X_bow.pkl")
joblib.dump(vectorizer, "D:/sentiment_analysis/data/temp_data/bow_vectorizer.pkl")

