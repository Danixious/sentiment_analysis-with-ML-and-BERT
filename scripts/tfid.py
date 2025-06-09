# import pandas as pd
# from sklearn.feature_extraction.text import CountVectorizer
# import joblib
# import ast
# from sklearn.feature_extraction.text import TfidfVectorizer

# data = pd.read_csv("D:/sentiment_analysis/data/cleaned_data.csv")

# def safe_join(x):
#     try:
#         if isinstance(x,str):
#             tokens = ast.literal_eval(x)
#             if isinstance(tokens,list):
#                 return ' '.join(tokens)
#     except Exception as e:
#         print(f"Error parsing: {e}")
#         return ''
#     return ''
# data['tf_word'] = data['text_lemmatize'].apply(safe_join)
# print("Empty rows before filtering:", sum(data['tf_word'].str.strip() == ''))
# data = data[data['tf_word'].str.strip() != '']
# print("Empty rows after filtering:", sum(data['tf_word'].str.strip() == ''))

# example = data['text_lemmatize'].iloc[0]
# tokens = ast.literal_eval(example) if isinstance(example, str) else []
# print("Tokens:", tokens)

# vectorizer = TfidfVectorizer()
# tf_word = vectorizer.fit_transform(data["tf_word"])

# print("Sample encoded : ",tf_word.toarray() )

# joblib.dump(tf_word,"D:/sentiment_analysis/data/temp_data/tfid_vectors.pkl")
# joblib.dump(vectorizer, "D:/sentiment_analysis/data/temp_data/tfid_vectorizer.pkl")