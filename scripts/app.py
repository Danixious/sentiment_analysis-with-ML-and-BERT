import streamlit as st
import pandas as pd
import joblib
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import plotly.express as px

# Label map
label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

# Load ML model and vectorizer
ml_model = joblib.load("D:/sentiment_analysis/data/temp_data/sentiment_model.pkl")
vectorizer = joblib.load("D:/sentiment_analysis/data/temp_data/bow_vectorizer.pkl")

# Load and cache BERT model + tokenizer
@st.cache_resource
def load_bert_model_and_tokenizer():
    tokenizer_path = "D:/sentiment_analysis/data/temp_data/bert_sentiment"
    model = BertForSequenceClassification.from_pretrained(tokenizer_path)
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    model.eval()
    return model, tokenizer

bert_model, bert_tokenizer = load_bert_model_and_tokenizer()

# App layout
st.title("🧠 Sentiment Analysis: ML vs BERT")
tab1, tab2 = st.tabs(["🤖 ML Model", "🔬 BERT Model"])

# ML Model Tab
with tab1:
    st.subheader("Machine Learning Model (Naive Bayes)")
    ml_mode = st.radio("Select ML Mode", ["Single Review", "Batch Review"])

    if ml_mode == "Single Review":
        text = st.text_area("✏️ Enter your review:")
        if st.button("Predict with ML"):
            X = vectorizer.transform([text])
            pred = ml_model.predict(X)[0]
            st.success(f"ML Sentiment: **{label_map[pred]}**")

    else:
        file = st.file_uploader("📁 Upload CSV for ML", type=["csv"], key="ml_file")
        if file:
            df = pd.read_csv(file)
            col = df.columns[0]
            X = vectorizer.transform(df[col].astype(str))
            preds = ml_model.predict(X)
            df["Sentiment"] = [label_map[p] for p in preds]
            st.dataframe(df)

            summary = df["Sentiment"].value_counts().reset_index()
            summary.columns = ["Sentiment", "Count"]
            summary["Percentage"] = round((summary["Count"] / summary["Count"].sum()) * 100, 2)

            st.write("### 📊 Sentiment Summary")
            df = df.loc[:,~df.columns.str.contains("^Unnamed")]
            st.dataframe(summary)
            st.plotly_chart(px.pie(summary, names="Sentiment", values="Count", title="ML Sentiment Distribution"))

# BERT Model Tab
with tab2:
    st.subheader("Transformer-Based Model (BERT)")
    bert_mode = st.radio("Select BERT Mode", ["Single Review", "Batch Review"])

    if bert_mode == "Single Review":
        text = st.text_area("✏️ Enter your review:",key = "bert_text_input")
        if st.button("Predict with BERT"):
            inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
            with torch.no_grad():
                outputs = bert_model(**inputs)
                pred = torch.argmax(outputs.logits, dim=1).item()
            st.success(f"BERT Sentiment: **{label_map[pred]}**")

    else:
        file = st.file_uploader("📁 Upload CSV for BERT", type=["csv"], key="bert_file")
        if file:
            df = pd.read_csv(file)
            col = df.columns[0]
            sentiments = []

            st.info("🔄 Predicting sentiments using BERT sit tight...")
            progress_bar = st.progress(0)
            for i, text in enumerate(df[col].astype(str)):
                inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
                with torch.no_grad():
                    outputs = bert_model(**inputs)
                    pred = torch.argmax(outputs.logits, dim=1).item()
                    sentiments.append(label_map[pred])

                # Update progress
                progress_bar.progress((i + 1) / len(df))

            df["Sentiment"] = sentiments
            st.success("✅ Prediction Complete!")
            st.dataframe(df)

            summary = df["Sentiment"].value_counts().reset_index()
            summary.columns = ["Sentiment", "Count"]
            summary["Percentage"] = round((summary["Count"] / summary["Count"].sum()) * 100, 2)

            st.write("### 📊 Sentiment Summary")
            st.dataframe(summary)
            st.plotly_chart(px.pie(summary, names="Sentiment", values="Count", title="BERT Sentiment Distribution"))
