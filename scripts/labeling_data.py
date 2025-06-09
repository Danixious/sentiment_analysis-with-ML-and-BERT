import pandas as pd
from textblob import TextBlob

# Load the cleaned dataset
data = pd.read_csv("D:/sentiment_analysis/data/data.csv",encoding = 'ISO-8859-1',on_bad_lines='skip')

# Check the initial shape and columns
print("Data loaded shape:", data.shape)
print("Columns:", data.columns.tolist())

# Function to label sentiments using TextBlob
def label_sentiment(text):
    if pd.isnull(text) or not isinstance(text,str) or text.strip() == "":
        return "Neutral"
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return "Positive"
    elif analysis.sentiment.polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# Apply the sentiment labeling function to the 'Text' column
data['Sentiment'] = data['Text'].apply(label_sentiment)

label_map = {"Positive": 2,"Neutral": 1,"Negative":0}
data["Sentiment_label"] = data["Sentiment"].map(label_map)

data.drop(columns = ['Review'],inplace = True,errors ='ignore')

# Check the distribution of the new Sentiment labels
print("\n Label value counts (including NaN):")
print(data['Sentiment'].value_counts(dropna=False))
print("\n Numeric label Distribution: ")
print(data['Sentiment_label'].value_counts(dropna = False))
# Save the labeled data to a new CSV file
labeled_data_path = "D:/sentiment_analysis/data/labeled_data.csv"
data.to_csv(labeled_data_path, index=False)

print(f"Labeled data has been saved to: {labeled_data_path}")

