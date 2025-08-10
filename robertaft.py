import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, get_scheduler
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm
import numpy as np

MODEL_NAME = "roberta-base"
BATCH_SIZE = 16
LR = 2e-5
EPOCHS = 4
MAX_LEN = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_path = os.path.dirname(os.path.abspath(__file__))
file1 = os.path.join(base_path, "Sarcasm_Headlines_Dataset.json")
file2 = os.path.join(base_path, "Sarcasm_Headlines_Dataset_v2.json")

df1 = pd.read_json(file1, lines=True)
df2 = pd.read_json(file2, lines=True)
df = pd.concat([df1, df2], ignore_index=True)
df = df[["headline", "is_sarcastic"]].rename(columns={"headline": "text", "is_sarcastic": "label"})

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text"].tolist(),
    df["label"].tolist(),
    test_size=0.1,
    stratify=df["label"],
    random_state=42
)

tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

class SarcasmDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        encodings = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": encodings["input_ids"].squeeze(0),
            "attention_mask": encodings["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

train_dataset = SarcasmDataset(train_texts, train_labels, tokenizer, MAX_LEN)
val_dataset = SarcasmDataset(val_texts, val_labels, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.to(DEVICE)

optimizer = AdamW(model.parameters(), lr=LR)
num_training_steps = EPOCHS * len(train_loader)
scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=500, num_training_steps=num_training_steps)

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} Training", leave=False)
    for batch in loop:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
        loop.set_postfix(loss=loss.item())
    avg_train_loss = train_loss / len(train_loader)
    print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f}")

    model.eval()
    preds, true_labels, probas = [], [], []
    loop = tqdm(val_loader, desc="Evaluating", leave=False)
    with torch.no_grad():
        for batch in loop:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)[:, 1]
            predictions = torch.argmax(logits, dim=-1)
            preds.extend(predictions.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            probas.extend(probabilities.cpu().numpy())

    print("\n=== Classification Report ===")
    print(classification_report(true_labels, preds, target_names=["Not Sarcastic", "Sarcastic"]))
    print(f"Accuracy: {accuracy_score(true_labels, preds):.4f}")
    print(f"Precision: {precision_score(true_labels, preds):.4f}")
    print(f"Recall: {recall_score(true_labels, preds):.4f}")
    print(f"F1 Score: {f1_score(true_labels, preds):.4f}")
    try:
        print(f"ROC AUC: {roc_auc_score(true_labels, probas):.4f}")
    except:
        print("ROC AUC could not be calculated.")
    print("\nConfusion Matrix:")
    print(confusion_matrix(true_labels, preds))

model.save_pretrained(os.path.join(base_path, "sarcasm_model"))
tokenizer.save_pretrained(os.path.join(base_path, "sarcasm_model"))
print(f"Model saved to {os.path.join(base_path, 'sarcasm_model')}")
