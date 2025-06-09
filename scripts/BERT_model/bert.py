import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
    AdamW
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # for progress bar
import joblib
# Hyperparameters
Model_Name = "bert-base-cased"
Max_Len = 128
Batch_size = 16
Epochs = 3
Lr = 2e-5
Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom Dataset
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=Max_Len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(label, dtype=torch.long)
        }

# Load and prepare data
data = pd.read_csv("D:/sentiment_analysis/data/cleaned_data_bert.csv")
data = data.dropna(subset=["Text", "Sentiment_label"])

print("📊 Dataset size:", len(data))

train_texts, val_texts, train_labels, val_labels = train_test_split(
    data["Text"].tolist(),
    data["Sentiment_label"].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=data["Sentiment_label"]
)

tokenizer = BertTokenizer.from_pretrained(Model_Name)

# DEBUG: show a tokenized sample
print("🔍 Sample Text:", train_texts[0])
print("🧪 Tokenized Sample:", tokenizer(train_texts[0]))

train_ds = SentimentDataset(train_texts, train_labels, tokenizer)
val_ds = SentimentDataset(val_texts, val_labels, tokenizer)

print("✅ Training samples:", len(train_ds))
print("✅ Validation samples:", len(val_ds))

train_loader = DataLoader(train_ds, batch_size=Batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=Batch_size)

print("📦 Total Training Batches:", len(train_loader))

# Initialize model
model = BertForSequenceClassification.from_pretrained(Model_Name, num_labels=3)
model.to(Device)

optimizer = AdamW(model.parameters(), lr=Lr)
total_steps = len(train_loader) * Epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Training Loop
for epoch in range(Epochs):
    model.train()
    total_train_loss = 0

    print(f"\n🚀 Starting Epoch {epoch + 1}/{Epochs}")
    progress_bar = tqdm(train_loader, desc="Training", leave=False)

    for i, batch in enumerate(progress_bar):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(Device)
        attention_mask = batch["attention_mask"].to(Device)
        labels = batch["label"].to(Device)  

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        total_train_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()

        progress_bar.set_description(f"Epoch {epoch + 1} | Loss: {loss.item():.4f}")

    avg_train_loss = total_train_loss / len(train_loader)
    print(f"✅ Epoch {epoch + 1} Completed — Avg Loss: {avg_train_loss:.4f}")

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(Device)
            attention_mask = batch["attention_mask"].to(Device)
            labels = batch["label"].to(Device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    print(f"🎯 Validation Accuracy: {correct / total:.4f}")

# Save Model
output_dir = "D:/sentiment_analysis/data/temp_data/bert_sentiment"
os.makedirs(output_dir, exist_ok=True)
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"✅ Model and tokenizer saved to {output_dir}")
