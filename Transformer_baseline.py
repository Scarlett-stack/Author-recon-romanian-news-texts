#!/usr/bin/env python
# coding: utf-8

# # Partea a treia. Transformer baseline
# - la nivel de text, titlu, propozitie, summary. (?)

# In[1]:


import pandas as pd
import re
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import torch.nn as nn
from transformers import AutoModel
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import f1_score
import torch
import numpy as np
import random

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib
matplotlib.use("Agg")

import os
os.makedirs("outputs", exist_ok=True)

# In[2]:


# FUNCTII UTILE le definim sus
def drop_columns(df, columns_to_drop):
    return df.drop(columns=columns_to_drop)

def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.replace("\n", " ")
    text = text.replace('""', '"')
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


# In[3]:


# citim
top20_ro_train_text = pd.read_csv('final_stilo/top20_ro_train.csv')
top20_ro_val_text = pd.read_csv('final_stilo/top20_ro_val.csv')
top20_ro_test_text = pd.read_csv('final_stilo/top20_ro_test.csv')


# In[4]:


# PAS 1: Label encoding autori
# folosim data loader pentru bert (ca la pytorch)
# autorii trebuie encodati pentru ca bert nu lucreaza cu stringuri ca etichete
l_encoder = LabelEncoder()
l_encoder.fit(top20_ro_train_text['author'])

y_train_t20ro = l_encoder.transform(top20_ro_train_text['author'])
y_val_t20ro = l_encoder.transform(top20_ro_val_text['author'])
y_test_t20ro = l_encoder.transform(top20_ro_test_text['author'])

num_authors = len(l_encoder.classes_)
print(f"Numar autori: {num_authors}")
print(f"Autori: {l_encoder.classes_}")



# In[5]:


# Pas 2: preprocesare text pt bert
# nu putem trimite textul cu mizerii
# curatam si textul si pastram doar asta din datafreames
# convertim in lista pentru siguranta in dataloader
texts_train_t20ro = top20_ro_train_text['text'].apply(clean_text).tolist()
texts_val_t20ro  = top20_ro_val_text['text'].apply(clean_text).tolist()
texts_test_t20ro  = top20_ro_test_text['text'].apply(clean_text).tolist()


# In[6]:


# Pas 4 dataset cu sliding window
tokenizer = AutoTokenizer.from_pretrained("dumitrescustefan/bert-base-romanian-cased-v1")

class AuthorDataset(Dataset):
    def __init__(self, texts, labels, tokenizer,
                 max_length=512, overlap=0.1):
        self.texts      = texts
        self.labels     = labels
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.stride     = int(max_length * (1 - overlap))  # 460


    def __len__(self):
        return len(self.texts)


    def __getitem__(self, idx):
        text  = self.texts[idx]
        label = self.labels[idx]

        # tokenizam fara truncation ca sa vedem lungimea reala
        tokens = self.tokenizer(
            text,
            truncation=False,
            add_special_tokens=True,
            return_tensors='pt'
        )
        input_ids = tokens['input_ids'][0]

        # daca e scurt - o singura fereastra
        if len(input_ids) <= self.max_length:
            encoded = self.tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            return {
                'input_ids':      encoded['input_ids'].squeeze(0),
                'attention_mask': encoded['attention_mask'].squeeze(0),
                'label':          torch.tensor(label, dtype=torch.long),
                'n_windows':      torch.tensor(1, dtype=torch.long)
            }

        # daca e lung - luam prima fereastra pentru training
        # (la inferenta folosim toate ferestrele cu voting)
        input_ids_window = input_ids[:self.max_length]
        attention_mask   = torch.ones(self.max_length, dtype=torch.long)

        return {
            'input_ids':      input_ids_window,
            'attention_mask': attention_mask,
            'label':          torch.tensor(label, dtype=torch.long),
            'n_windows':      torch.tensor(
                (len(input_ids) // self.stride) + 1,
                dtype=torch.long
            )
        }


# In[11]:


# Pas 4 cream dataloaderle
BATCH_SIZE = 8  # mic din cauza RAM GPU

# si aici la random shuffle trebuie activat seed
generator = torch.Generator()
generator.manual_seed(42)

train_dataset = AuthorDataset(texts_train_t20ro, y_train_t20ro, tokenizer)
val_dataset   = AuthorDataset(texts_val_t20ro,   y_val_t20ro,   tokenizer)
test_dataset  = AuthorDataset(texts_test_t20ro,  y_test_t20ro,  tokenizer)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    generator=generator
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    generator=generator
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    generator=generator
)

print(f"Train batches: {len(train_loader)}")
print(f"Val batches:   {len(val_loader)}")


# In[12]:


# Pas 5 Modelul efectiv si forward pass
class BaselineBERT(nn.Module):
    def __init__(self, num_authors, dropout=0.2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(
            "dumitrescustefan/bert-base-romanian-cased-v1"
        )
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, num_authors)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls_output = outputs.last_hidden_state[:, 0, :]  # CLS token
        cls_output = self.dropout(cls_output)
        logits     = self.classifier(cls_output)
        return logits


# In[13]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device {device}")


# In[14]:


batch = next(iter(train_loader))
print(batch['input_ids'][0][:10])  # primii 10 tokens din primul exemplu
# tensor([    2,  3425,    17,    83,  8897,   363,  3754, 14002,   695,   398])


# In[15]:


model = BaselineBERT(num_authors=num_authors).to(device)


# In[16]:


# asta o pun sus
EPOCHS      = 3
LR          = 2e-5
WARMUP_STEPS = len(train_loader) // 10  # 10% din train


# In[17]:


optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=WARMUP_STEPS,
    num_training_steps=len(train_loader) * EPOCHS
)
criterion = nn.CrossEntropyLoss()


# In[18]:


def train_epoch(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0.0

    for i, batch in enumerate(loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        # init gradienti cu 0
        optimizer.zero_grad(set_to_none=True) # setam pe None ca e mai rapid
        logits = model(input_ids, attention_mask)
        loss = criterion(logits,labels)

        # asta a fost forward pass
        # acum backprop ca sa setam weights
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        if i % 100 == 0:
            print(f"  Batch {i}/{len(loader)} loss: {loss.item():.4f}")

    return total_loss / len(loader)

def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['label'].to(device)

            logits = model(input_ids, attention_mask)
            loss   = criterion(logits, labels)

            preds = torch.argmax(logits, dim=-1)

            total_loss  += loss.item()
            all_preds   .extend(preds.cpu().numpy())
            all_labels  .extend(labels.cpu().numpy())

    f1 = f1_score(all_labels, all_preds, average='macro')
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    return total_loss / len(loader), f1, accuracy


# In[19]:


# training loop
best_f1 = 0
train_losses = []
val_losses = []
val_f1_scores = []
val_accuracies = []

train_f1_scores = []
train_accuracies = []

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    train_loss = train_epoch(
        model, train_loader, optimizer, scheduler, criterion, device
    )
    # scoatem scorurile din epoca respectiva
    _, train_f1, train_acc = eval_epoch(model, train_loader, criterion, device)
    train_losses.append(train_loss)
    train_f1_scores.append(train_f1)
    train_accuracies.append(train_acc)

    val_loss, val_f1, val_acc = eval_epoch(model, val_loader, criterion, device)
    val_losses.append(val_loss)
    val_f1_scores.append(val_f1)
    val_accuracies.append(val_acc)

    print(f"Train loss: {train_loss:.4f} | Train F1: {train_f1:.4f} | Train Accuracy: {train_acc:.4f}")
    print(f"Val loss:   {val_loss:.4f}  |  Val F1: {val_f1:.4f} | Val Accuracy: {val_acc:.4f}")

    # salvezi cel mai bun model
    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), 'outputs/best_model_ro_top20.pt')
        print(f"  Model salvat (F1={val_f1:.4f})")

print(f"\nBest Val F1: {best_f1:.4f}")
# 155 minute F1 la a treia epoca 0.6654 !!!

# Epoch 1/4
#   Batch 0/3106 loss: 3.1662
#   Batch 100/3106 loss: 2.8325
#   Batch 200/3106 loss: 2.8588
# cu seed si aia de dnn pe true 177 


# ### Training Progress Plots

# In[20]:
with open("outputs/metrics.txt", "w") as f:
    f.write(f"Best Val F1: {best_f1}\n")
    f.write(f"Final Val Accuracy: {val_accuracies[-1]}\n")

epochs_range = range(1, EPOCHS + 1)

plt.figure(figsize=(18, 5))

# Plot Loss
plt.subplot(1, 3, 1)
plt.plot(epochs_range, train_losses, label='Train Loss')
plt.plot(epochs_range, val_losses, label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Plot F1 Score
plt.subplot(1, 3, 2) 
plt.plot(epochs_range, train_f1_scores, label='Train F1 Score', color='blue')
plt.plot(epochs_range, val_f1_scores, label='Validation F1 Score', color='orange')
plt.title('F1 Score over Epochs')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.legend()
plt.grid(True)

# Plot Accuracy
plt.subplot(1, 3, 3) 
plt.plot(epochs_range, train_accuracies, label='Train Accuracy', color='purple')
plt.plot(epochs_range, val_accuracies, label='Validation Accuracy', color='green')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("outputs/training_curves.png")
plt.close()

# In[21]:


# Pas5 Majority vote pe setul de test
def predict_with_voting(model, text, tokenizer,
                         max_length=512, overlap=0.1, device='cuda'):
    stride = int(max_length * (1 - overlap))
    model.eval()

    tokens = tokenizer(
        text, truncation=False,
        add_special_tokens=True, return_tensors='pt'
    )
    input_ids = tokens['input_ids'][0]

    predictions = []

    with torch.no_grad():
        for start in range(0, len(input_ids), stride):
            end      = min(start + max_length, len(input_ids))
            window   = input_ids[start:end]

            # padding
            if len(window) < max_length:
                padding = torch.zeros(max_length - len(window), dtype=torch.long)
                window  = torch.cat([window, padding])

            attn_mask = (window != 0).long()

            logits = model(
                window.unsqueeze(0).to(device),
                attn_mask.unsqueeze(0).to(device)
            )
            pred = torch.argmax(logits, dim=-1).item()
            predictions.append(pred)

            if end == len(input_ids):
                break

    from collections import Counter
    return Counter(predictions).most_common(1)[0][0]

# evaluezi pe test cu voting
model.load_state_dict(torch.load('outputs/best_model_ro_top20.pt'))

all_preds  = []
all_labels = []

for idx in range(len(texts_test_t20ro)):
    pred  = predict_with_voting(model, texts_test_t20ro[idx], tokenizer, device=device)
    all_preds.append(pred)
    all_labels.append(y_test_t20ro[idx])

print(classification_report(
    all_labels, all_preds,
    target_names=l_encoder.classes_
))

df_preds = pd.DataFrame({
    "true": all_labels,
    "pred": all_preds
})
df_preds.to_csv("outputs/test_predictions.csv", index=False)
# In[ ]:


# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=l_encoder.classes_,
            yticklabels=l_encoder.classes_,
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix - Test Set')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png")
plt.close()

# In[ ]:




