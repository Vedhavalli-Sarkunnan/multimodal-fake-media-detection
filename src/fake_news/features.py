import numpy as np
import torch
from transformers import RobertaTokenizer, RobertaModel
import os

MODEL_DIR = "models/roberta_finetuned"
CLASSIFIER_PATH = os.path.join(MODEL_DIR, "classifier_head.pt")

def roberta_vectorize(text_series, device, tokenizer, model, batch_size=16, max_length=128):

    embeddings = []
    model.eval()
    for i in range(0, len(text_series), batch_size):
        encoded = tokenizer(
            text_series[i:i+batch_size].to_list(),
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
    
        encoded = {key: val.to(device) for key, val in encoded.items()}

        with torch.no_grad():
            output = model(**encoded)
            
            # output.last_hidden_state shape:
            # (batch_size, sequence_length, hidden_size)
            # hidden_size = 768
            cls_embedding = output.last_hidden_state[:,0,:] # <s> token
            embeddings.append(cls_embedding.cpu().numpy())

    return np.vstack(embeddings)

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.metrics import f1_score

class FakeNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }

class RobertaFeatureTuner(nn.Module):
    def __init__(self, roberta_model, num_classes=2):
        super().__init__()

        self.roberta = roberta_model
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
         # output.last_hidden_state shape:
         # (batch_size, sequence_length, hidden_size)
         # hidden_size = 768
         # : → all samples, 0 → CLS token, : → all 768 features
        
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        cls_embedding = self.dropout(cls_embedding)
        
        logits = self.classifier(cls_embedding)
        return logits, cls_embedding

def fine_tune_roberta(
    texts_train,
    labels_train,
    texts_val,
    labels_val,
    epochs=3,
    batch_size=16,
    lr=2e-5
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    train_ds = FakeNewsDataset(texts_train, labels_train, tokenizer)
    val_ds = FakeNewsDataset(texts_val, labels_val, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    base_roberta = RobertaModel.from_pretrained("roberta-base").to(device)
    model = RobertaFeatureTuner(base_roberta).to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits, _ = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        preds, truths = [], []

        with torch.no_grad():
            for batch in val_loader:
                
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                
                logits, _ = model(input_ids, attention_mask)
                pred = torch.argmax(logits, dim=1)

                preds.extend(pred.cpu().numpy())
                truths.extend(labels.cpu().numpy())
                
        val_f1 = f1_score(truths, preds)
        print(f"Epoch {epoch+1} | Val F1: {val_f1:.4f}")
        
    for param in model.parameters():
        param.requires_grad = False

    print("Fine-tuning completed")
    return model, tokenizer

def extract_embeddings(texts, labels, model, tokenizer, batch_size=32, max_len=256):
    
    device = next(model.parameters()).device
    model.eval()
    
    embeddings = []
    true_labels = []

    dataset = FakeNewsDataset(texts, labels, tokenizer, max_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            _, cls_emb = model(input_ids, attention_mask)

            embeddings.append(cls_emb.cpu())
            true_labels.extend(batch["label"].cpu().numpy())
            
    embeddings = torch.cat(embeddings, dim=0).numpy()
    true_labels = np.array(true_labels)
    print("Extracted embeddings")
    return embeddings, true_labels

def save_finetuned_roberta(model, tokenizer, save_dir=MODEL_DIR):
    os.makedirs(save_dir, exist_ok=True)

    tokenizer.save_pretrained(save_dir)
    model.roberta.save_pretrained(save_dir)

    torch.save(
        model.classifier.state_dict(),
        CLASSIFIER_PATH
    )
    print("Fine-tuned RoBERTa saved")

def load_finetuned_roberta(device, save_dir=MODEL_DIR):
    tokenizer = RobertaTokenizer.from_pretrained(save_dir)
    roberta = RobertaModel.from_pretrained(save_dir)

    model = RobertaFeatureTuner(roberta)
    model.classifier.load_state_dict(
        torch.load(CLASSIFIER_PATH, map_location=device)
    )

    model.to(device)
    model.eval()

    print("Fine-tuned RoBERTa loaded")
    return model, tokenizer

def save_embeddings(embeddings, base_dir, dataset, split, modality=None):
    """
    embeddings : np.ndarray
    base_dir  : root directory (e.g. 'embeddings')
    dataset   : 'full', 'title_only', 'body_only'
    split     : 'train', 'val', 'test'
    modality  : 'title' or 'body' (only for full dataset)
    """

    if modality:
        save_dir = os.path.join(base_dir, dataset, split)
        filename = f"{modality}.npy"
    else:
        save_dir = os.path.join(base_dir, dataset)
        filename = f"{split}.npy"

    os.makedirs(save_dir, exist_ok=True)

    path = os.path.join(save_dir, filename)
    np.save(path, embeddings)

    print(f"Saved: {path}")

def load_embeddings(base_dir, dataset, split, modality=None):
    if modality:
        path = os.path.join(base_dir, dataset, split, f"{modality}.npy")
    else:
        path = os.path.join(base_dir, dataset, f"{split}.npy")

    return np.load(path)