#!/usr/bin/env python
# coding: utf-8

# # Partea 4.2. Contrastive cu triplet loss si hard negative mining
# 
# - in loc sa mai calculam matricile de perechi , ne folosim de 1 anchor example, 1 pozitive example, si 1 negative example si calculam triplet loss ca in : (link)[https://lilianweng.github.io/posts/2021-05-31-contrastive/]
# 
# - dataset -> windows (cu overlap)
# - ContrastiveExamplesManager (pentru negative inteligente)
# - model = BERT + projection head
# - dataloader care produce triplets
# - training: CrossEntropy + TripletLoss

# In[ ]:


# !pip install transformers sentence-transformers scikit-learn


# In[1]:


import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import re
import unicodedata
from torch.utils.data import Sampler
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import json
import random
from collections import Counter

import seaborn as sns

# fara optimizari avem 7gb ram ocupati 
# o sa zica ca torch,cuda.amp e deprecated dar pe fep nu merge altfel
from torch.cuda.amp import autocast, GradScaler


# ### Transformer backbone
# - pentru antrenare vom folosi backbone-ul obsinuit de BERT  (fine tuned cu parametrii de pana acum) , dar user embeddings le vom extrage cu un Sentence TRansformer (deocamdata am unul tot bert based urmeaza sa il testam un pic). 
# - ideea e sa scoatem embeddinguri care sa surprinda cat mai mult din stil si dupa cum am vazut la expermientul contrastie anterior, BERT nu reuseste foarte bine sa scoata nici cu mean pooling nici cu CLS token. Adica reuseste intr-o oarecare masura dar putem imbunatati ? 

# In[2]:


# definim aici variabilele globale
BERT_MODEL = "dumitrescustefan/bert-base-romanian-cased-v1"
ST_MODEL = "iliemihai/sentence-bert-base-romanian-uncased-v1"


# In[3]:


MAX_LEN = 512
CONTRASTIVE_WEIGHT = 0.3
OVERLAP = 0.3
LEARNING_RATE = 2e-5
OUTPUT_DIR = 'outputs_contrastive_2'
tokenizer = AutoTokenizer.from_pretrained("dumitrescustefan/bert-base-romanian-cased-v1")


# In[4]:


# normalizare sedila in virgual conform documentatiei de pe hugging face
def normalize_romanian(text):
    return text.replace("ţ", "ț").replace("ş", "ș") \
               .replace("Ţ", "Ț").replace("Ş", "Ș")


# - testare exemplu documentatie Sentence bert:
# - intre timp s-a facut update la pytorch si a aparut versiunea 2.6 noi avem 2.5 dar , dupa cum am vazut pe cluster o sa prefer sa raman la versiunea de 2.5 pentru a evita erori aiurea

# In[5]:


print(torch.__version__)


# In[6]:


# Inițializăm modelul
model = SentenceTransformer("iliemihai/sentence-bert-base-romanian-uncased-v1")

# Definim propozițiile
sentences = [
    "Un tren își începe călătoria către destinație.",
    "O locomotivă pornește zgomotos spre o stație îndepărtată.",
    "Un muzician cântă la un saxofon impresionant.",
    "Un saxofonist evocă melodii suave sub lumina lunii.",
    "O bucătăreasă presară condimente pe un platou cu legume.",
    "Un chef adaugă un strop de mirodenii peste o salată colorată.",
    "Un jongler aruncă si prinde mingi colorate.",
    "Un artist de circ jonglează cu măiestrie sub reflectoare.",
    "Un artist pictează un peisaj minunat pe o pânză albă.",
    "Un pictor redă frumusețea naturii pe pânza sa strălucitoare."
]

# Obținem embeddings pentru fiecare propoziție
embeddings = model.encode(sentences)

# Calculăm similaritatea semantică folosind similaritatea cosine
similarities = np.dot(embeddings, embeddings.T) / (np.linalg.norm(embeddings, axis=1)[:, np.newaxis] * np.linalg.norm(embeddings, axis=1)[np.newaxis, :])

# Identificăm cea mai similară propoziție pentru fiecare propoziție, excluzând similaritatea cu sine însăși
most_similar_indices = np.argmax(similarities - np.eye(len(sentences)), axis=1)

most_similar_sentences = [(sentences[i], sentences[most_similar_indices[i]], similarities[i, most_similar_indices[i]]) for i in range(len(sentences))]

print(most_similar_sentences)


# In[7]:


# the usual functions
# FUNCTII UTILE le definim sus
def drop_columns(df, columns_to_drop):
    return df.drop(columns=columns_to_drop)

# avem o problema la moldova
# diacriticele aparent nu sunt unicode?
def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFC", text)

    # sedila -> virgula
    text = normalize_romanian(text)

    # remove combining chars, deci t de la tap nu va mai avea caciulita sepaarata va fi un singur char
    text = ''.join(c for c in text if not unicodedata.combining(c))

    text = text.replace("„", '"').replace("”", '"')
    text = text.replace("’", "'")

    # remove control chars
    text = ''.join(c for c in text if unicodedata.category(c)[0] != 'C')

    text = text.replace("\n", " ")
    text = text.replace("\t", " ")
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


# #### Grafice:

# In[8]:


def plot_training_curves(train_losses, val_losses,
                          train_f1s, val_f1s,
                          train_accs, val_accs,
                          dataset_name="", out_dir="."):
    epochs_range = range(1, len(train_losses) + 1)

    plt.figure(figsize=(18, 6))
    plt.suptitle(f'Training Progress — {dataset_name}', fontsize=14, y=1.02)

    # Loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, train_losses, label='Train Loss', marker='o', color='steelblue')
    plt.plot(epochs_range, val_losses,   label='Val Loss',   marker='o', color='coral')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # F1
    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, train_f1s, label='Train F1', marker='o', color='blue')
    plt.plot(epochs_range, val_f1s,   label='Val F1',   marker='o', color='orange')
    plt.title('Macro F1 over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)

    # Accuracy
    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, train_accs, label='Train Accuracy', marker='o', color='purple')
    plt.plot(epochs_range, val_accs,   label='Val Accuracy',   marker='o', color='green')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'training_curves_contrastive2_{dataset_name}.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(y_true, y_pred, label_encoder,
                              dataset_name="", split='test', out_dir="."):

    labels = label_encoder.classes_
    n = len(labels)

    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig_size = max(12, n * 0.8)

    font_size = max(6, int(300 / n))

    # === COUNTS ===
    plt.figure(figsize=(fig_size, fig_size))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        annot_kws={"size": font_size},
        square=True,
        cbar=True
    )

    plt.title(f'Confusion Matrix (Counts) — {dataset_name} | {split}', fontsize=16)
    plt.xlabel('Predicted')
    plt.ylabel('True')

    plt.xticks(rotation=45, ha='right', fontsize=font_size)
    plt.yticks(rotation=0, fontsize=font_size)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'cm_counts_contrastive2_{dataset_name}_{split}.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # === NORMALIZED ===
    plt.figure(figsize=(fig_size, fig_size))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        annot_kws={"size": font_size},
        square=True,
        cbar=True,
        vmin=0,
        vmax=1
    )

    plt.title(f'Confusion Matrix (Normalized) — {dataset_name} | {split}', fontsize=16)
    plt.xlabel('Predicted')
    plt.ylabel('True')

    plt.xticks(rotation=45, ha='right', fontsize=font_size)
    plt.yticks(rotation=0, fontsize=font_size)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'cm_norm_contrastive2_{dataset_name}_{split}.png'), dpi=150, bbox_inches='tight')
    plt.close()


# Adaugam si un grafic in care vizualizam cum scade fiecare compinenta a loss-ului total: CE si Triplet

# In[9]:


def plot_contrastive_losses(train_ce_losses,
                            train_nce_losses,
                            contrastive_weight=CONTRASTIVE_WEIGHT,
                            dataset_name="",
                            output_dir="."):

    epochs_range = range(1, len(train_ce_losses) + 1)

    weighted_nce = [
        contrastive_weight * x
        for x in train_nce_losses
    ]

    total_loss = [
        ce + wnce
        for ce, wnce in zip(train_ce_losses, weighted_nce)
    ]

    plt.figure(figsize=(10, 6))

    # CE
    plt.plot(
        epochs_range,
        train_ce_losses,
        marker='o',
        label='CrossEntropy Loss'
    )

    # raw InfoNCE
    plt.plot(
        epochs_range,
        train_nce_losses,
        marker='o',
        label='Raw InfoNCE Loss'
    )

    # weighted InfoNCE
    plt.plot(
        epochs_range,
        weighted_nce,
        marker='o',
        label=f'Weighted InfoNCE (x{contrastive_weight})'
    )

    # total
    plt.plot(
        epochs_range,
        total_loss,
        marker='o',
        linestyle='--',
        label='Total Loss'
    )

    plt.title(f'Contrastive Loss Components — {dataset_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_dir,
            f'contrastive2_losses_{dataset_name}.png'
        ),
        dpi=150,
        bbox_inches='tight'
    )

    plt.close()


# ### Pas 0: clasele de pana acum si functia de calculat ferestre

# In[10]:


class AuthorDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=MAX_LEN):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids':      encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'label':          torch.tensor(self.labels[idx], dtype=torch.long)
        }


# In[11]:


def split_into_windows(text, tokenizer, max_length=MAX_LEN, overlap=OVERLAP):
    # permitem si overlap
    # returnam o lista de stringuri (ferestre)
    stride = int(max_length * (1-overlap)) # cu asta parcurg ~ step
    # scoatem tokens cu tokenizer
    tokens = tokenizer(
        text,
        truncation=False, # pt windows
        add_special_tokens=False,
        return_tensors='pt' # pytorch tensors buni pt cuda
    )
    input_ids = tokens['input_ids'][0]

    # daca e mai mic ca max_len il returnam direct asa
    if len(input_ids) <= max_length:
        # return [text] # nu prea e ok asa ca nu e tokenizat?
        return [
        tokenizer.decode(
        input_ids,
        skip_special_tokens=True)]

    # impartim in ferestre
    windows = []
    for start in range(0, len(input_ids), stride):
        # iau end
        end = min(len(input_ids), start + max_length)
        window_ids = input_ids[start:end]
        # decodam pt compatibilitate cu ce o sa faca bert
        window_text = tokenizer.decode(window_ids, skip_special_tokens=True)
        windows.append(window_text)
        if end == len(input_ids):
            break

    return windows


def prepare_texts_with_windows(texts, labels, tokenizer, max_length=MAX_LEN,overlap=OVERLAP):
    # acum mapam autori -> [win1, win2 ...]
    # returnam liste extinse de texte si etichete
    all_texts = []
    all_labels = []

    for text, label in zip(texts, labels):
        windows = split_into_windows(text, tokenizer, max_length, overlap)
        all_texts.extend(windows)
        all_labels.extend([label] * len(windows))

    counts = Counter(all_labels)
    print(f"Texte originale: {len(texts)}")
    print(f"Ferestre totale: {len(all_texts)}")
    for label, count in sorted(counts.items()):
        print(f"  autor {label}: {count} ferestre")

    return all_texts, all_labels


# ### Pas 1: Backbone BERT 
# - modelul are dual objective: clasificare (`self.classifier`) directa si calcul de embeddings (`self.projection`) adica textele aceluias autor se vor afla in proximitate in spatiul vectorial spre deosebire de cele scrise de alt autor
# 
# 

# In[12]:


class TripleBERTAuthorship(nn.Module):
    def __init__(self, num_authors, projection_dim=128, dropout=0.2):
        # definim starturile aici 
        """
        INIT:
            load BERT
            create classifier head
            create projection head
        """
        super().__init__()  # apelam constructorul 
        self.bert = AutoModel.from_pretrained(BERT_MODEL) # backbone

        # PUTEM ADAUGA CHECKPOINT
        self.bert.gradient_checkpointing_enable()

        # ce primeste bert?
        # input_ids      : (batch_size, seq_len) ex (8, 512) = 8 texte cu 512 tokeni ?
        # attention_mask : (batch_size, seq_len)
        # ce output are?
        # bert.last_hidden_state adica:
        # (batch_size, seq_len, hidden_size) (8, 512, 768)
        self.dropout = nn.Dropout(dropout)

        # classifier head din embedding (batch_size, 768) scot predictie -> (batch_size, num_authors)
        self.classifier = nn.Linear(768, num_authors)

        # projection head simplificat nu mai am start intermediar de 256:
        # dar cred ca o sa pun strat intermediar vedem
        self.projection_head = nn.Sequential(
            nn.Linear(768, projection_dim)
        )
        # il folosim pt triplet loss, cosine similarity si info nce


    def forward(self, input_ids, attention_mask):
        """
        FORWARD:
            trece textul prin BERT
            ia tokenul CLS
            aplică dropout

            clasificare:
                CLS -> logits

            embeddings:
                CLS -> projection head
                normalize embeddings

            return logits + embeddings
        """
        # scoatem outputul de la bert
        bert_out = self.bert(
            input_ids = input_ids,
            attention_mask = attention_mask
        )
        # ne trebuie cls token -> embedding global
        # luam toate textele, tokenul 0 [CLS cel care rezuma], si toate cele 768 features
        cls_token = bert_out.last_hidden_state[:, 0, :]

        cls_token = self.dropout(cls_token)
        # clasificare
        logits = self.classifier(cls_token) # (batch_size, num_authors)

        # pe langa asta facem si projection pentru user embeddings
        projection = self.projection_head(cls_token) # (batch_size, num_authors)

        # normalizam 
        projection = F.normalize(projection, p=2, dim=1) # p e exponentul din formula de normalizare default e 2 oricum
        return logits, projection


    def get_embedding(self, input_ids, attention_mask):
        # il folosim pentru infrenta, embeding retrival si cosine similarity 
        # nu clasificam nimic aici , doar scoatem embedding
        with torch.no_grad():
            bert_out = self.bert(
                input_ids = input_ids,
                attention_mask = attention_mask
            )

            cls = bert_out.last_hidden_state[:, 0, :]
            projection = self.projection_head(cls)
            projection = F.normalize(projection,p=2, dim=1) # tensor

        return projection


# ### Pas 2:
# - author embeddings folosind `Sentence Bert` pentru a capta mai bine stilul unui autor
# - acesta este un model preantrenat pentru semantic similarity 
# - Aceste embeddinguri vor fi folosite doar pentru calcularea de hard negatives 

# In[13]:


# mai intai cream un vector user embedding global pt fiecare autor
def compute_author_profiles(sentence_model, texts_train_windows, y_train_windows, label_encoder, max_texts_per_auth, batch_size=64):
    """
    pentru fiecare autor:
        find toate ferestrele lui
        keep maxim K ferestre
        encode ferestrele cu Sentence-BERT
        get embedding pentru fiecare fereastră
        cpmpute media embeddingurilor
        save embeddingul mediu al autorului
    """
    author_profiles = {} # aici tinem embeddingurile <label> : tensor

    for author_idx, author_label in enumerate(label_encoder.classes_):
        # scoatem indicii autorului curent
        indices = [ i for i, y in enumerate(y_train_windows) if y == author_idx]
        # o sa luam o parte din texte
        indices = indices[:max_texts_per_auth]
        author_texts = [texts_train_windows[i] for i in indices] # [text1, text2 ...]

        # calculam embeddingurile textelor curente
        embeddings = sentence_model.encode(
            author_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_tensor=True
        )

        # facem media peste ce obtinem si le bagam in dictionar
        author_profiles[author_idx] = embeddings.mean(dim=0).cpu()
        print(f"{author_label}: {len(indices)} ferestre encodate")
    return author_profiles


# ### Pas 3: Hard Negative Mining 
# - aici ne uitam la toate embedingurile autorilor si calcuma distanta dintre ele
# - am vazut la expermientul anterior ca random sampling de texte nu functioneaza tocmai ideal
# - aici supervizam mai atent procesul de selectie al sample-urilor 
# - In final o sa avem : 
# `{
#    autor_A: [autori_negativi],
#    autor_B: [autori_negativi],
# }`
# 

# In[14]:


def compute_hard_negatives(author_profiles, label_encoder, top_k=5):
    # facem o lista sortat a autorilor
    authors_list = sorted(author_profiles.keys())

    # cream matricea cu toti autorii
    # ca la info nce inmutlim cu transpusa si vedem cst de similari sunt autorii intre eo
    # hard negatives se refera la autorii similari intre ei adica unde poate gresi usor modelul
    profiles_matrix = torch.stack([author_profiles[a] for a in authors_list])

    profiles_norm = F.normalize(profiles_matrix, p=2, dim=1)
    sim_matrix = profiles_norm @ profiles_norm.T

    hard_negatives = {}

    for i , author in enumerate(authors_list):
        similarities = sim_matrix[i].clone()
        # excludem autorul insusi
        similarities[i] = -2.0

        # cei mai similari -> hard negatives
        top_k_indexes = torch.topk(similarities, top_k, largest = True).indices.tolist()

        hard_negatives[author] = [ authors_list[j] for j in top_k_indexes]

        author_name = label_encoder.classes_[author]
        hard_negatives_names =[label_encoder.classes_[j] for j in hard_negatives[author]]

        print(f"{author_name} - negatives: {hard_negatives_names}")

    return hard_negatives


# ### Pas 4: Construire Dataset pentru triplete
# 
# - daca initial am lucrat cu (text, autor) acum trebuie sa lucram cu (anchor, positive, negative)
# - ideea este ca triplet loss vrea ca `anchor` sa fie cat mai apropiat de `positive` si cat mai deparat de `negative`
# - vom folosi un generator random rng calculat in functie de epoca astfel: `seed + epoch` , pentru stabilitatea rezultatelor si reproductibilitatea experimentului.
# - in caz ca nu sunt suficiente hard negatives luam random din dataset de la alt autor

# In[15]:


class TripleAuthorDataset(Dataset):
    def __init__(self, texts, labels, hard_negatives,  samples_per_author, epoch=0, seed=42):
        # cream generatorul random
        #rng = random.Random(seed + epoch)
        self.seed = seed
        self.epoch = epoch

        # construim dictionar autori: [lista texte/ferestre]
        self.author_texts ={}

        for text, label in zip(texts, labels):
            if label not in self.author_texts:
                self.author_texts[label] = []
            self.author_texts[label].append(text)

        self.valid_authors = sorted([a for a, t in self.author_texts.items() if len(t) >= 2])

        self.hard_negatives = hard_negatives

        # cate samples vrem per autor
        self.samples_per_author = samples_per_author
        print(f"Autori valid: {len(self.valid_authors)}")
        print(f"Samples per author: {self.samples_per_author}")



    def __len__(self):
        # return len(self.triplets)
        return len (self.valid_authors) * self.samples_per_author

    def __getitem__(self, idx):
        # anchor, positive, negative = self.triplets[idx]
        # label = self.triplet_labels[idx]

        rng = random.Random(self.seed + self.epoch*100000 + idx)

        # alegem autorul
        author = self.valid_authors[idx % len(self.valid_authors)]

        windows = self.author_texts[author]

        # anchor + positve
        anchor, positive = rng.sample(windows, 2) # luam 2 texte random
        # hard negative
        neg_authors = self.hard_negatives.get(author, [])

        # fallback in vcaz ca nu am desi sper sa nu ajunga aici
        if not neg_authors:
            neg_authors = [a for a in self.valid_authors if a != author]

        negative_author = rng.choice(neg_authors)
        # si luam textul lui

        negative = rng.choice(self.author_texts[negative_author])
        return {
            'anchor': anchor,
            'positive': positive,
            'negative': negative,
            'label': author
        }


# ### Pas 5: Collate fn si Dataloaderele
# - din moment ce folosim random in get item nu prea mai avem nevoie de unique sampler, pentru ca noi nu mai avem indexurile si tripletele hardcodate
# - habarnam daca o sa functioneze chestia asta , in cel mai rau caz refac cu triplete hardcodate in datset si aia e
# 
# Mica explicatie: dataset poate returna asa:  <br>
# **dataset[0]** <br>
# `{
#    anchor:   text_E1,
#    positive: text_E2,
#    negative: text_C1,
#    label: 0
# }`
# 
# 
# **dataset[1]**: 
# <br>
# 
# `{
#    anchor:   text_C2,
#    positive: text_C3,
#    negative: text_S1,
#    label: 1
# }
# `
# <br>
# 
# **Collate** primeste: 
# `batch = [
#    dataset[0],
#    dataset[1],
#    dataset[2],
#    ...
# ]`
# 
# - Label e pentru positive/anchor example!

# In[16]:


def triplet_collate_fn(batch):

    anchors = [item['anchor'] for item in batch]
    positives = [item['positive'] for item in batch]
    negatives =[item['negative'] for item in batch]
    labels = [item['label'] for item in batch] # am mai multe labels diferite?? 

    enc_a = tokenizer(anchors, max_length=MAX_LEN, truncation=True, padding='max_length', return_tensors='pt')
    enc_p = tokenizer(positives, max_length=MAX_LEN, truncation=True, padding='max_length', return_tensors='pt')
    enc_n = tokenizer(negatives, max_length=MAX_LEN, truncation=True, padding='max_length', return_tensors='pt')

    return {
        'input_ids_a': enc_a['input_ids'],
        'attention_mask_a': enc_a['attention_mask'],

        'input_ids_p': enc_p['input_ids'],
        'attention_mask_p': enc_p['attention_mask'],

        'input_ids_n': enc_n['input_ids'],
        'attention_mask_n': enc_n['attention_mask'],

        'label': torch.tensor(
            labels,
            dtype=torch.long
        )  # nu prea inteleg aici, sunt mai multe labels?

    }


# In[17]:


def make_triplet_loader(texts_windows, y_windows, hard_negatives, samples_per_author, batch_size, epoch, seed):
    # in principiu doar pe train?
    # ok ca sa garanta ca ordinea batch-urilor este aceasi la fiecare rulare facem cu generator
    g = torch.Generator()
    g.manual_seed(seed + epoch)
    dataset = TripleAuthorDataset(texts_windows, y_windows,
    hard_negatives=hard_negatives, samples_per_author=samples_per_author, epoch=epoch, seed=seed)

    loader = DataLoader(dataset,batch_size, shuffle=True, collate_fn=triplet_collate_fn,num_workers=0, pin_memory=True, generator=g)

    return loader


# Tot aici putem adauga si build_data_loader pentru val si text , si in asta curatam toate textele si il spargem in ferestre pe cele din train

# In[18]:


def build_dataloaders(train_df, val_df, test_df,
                               tokenizer, overlap=OVERLAP, seed=42):
    le = LabelEncoder()
    le.fit(train_df['author'])

    y_train = le.transform(train_df['author'])
    y_val   = le.transform(val_df['author'])
    y_test  = le.transform(test_df['author'])

    texts_train = train_df['text'].apply(clean_text).tolist()
    texts_val   = val_df['text'].apply(clean_text).tolist()
    texts_test  = test_df['text'].apply(clean_text).tolist()

    # overlap 0.3 ca studentul
    texts_train_w, y_train_w = prepare_texts_with_windows(
        texts_train, y_train, tokenizer, overlap=overlap
    )

    val_ds  = AuthorDataset(texts_val,  y_val,  tokenizer)
    test_ds = AuthorDataset(texts_test, y_test, tokenizer)

    val_loader  = DataLoader(val_ds,  batch_size=8,
                              shuffle=False, num_workers=0,
                              pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=8,
                              shuffle=False, num_workers=0,
                              pin_memory=True)
    return (le,
            texts_train_w, y_train_w,
            texts_val,     y_val,
            texts_test,    y_test,
            val_loader,    test_loader)


# ### Pas 6. Definirea functiei de loss si implementare train epoch
# - Daca cautam atent exista documentie pytorch! [triplet_margin_loss](https://docs.pytorch.org/docs/2.11/generated/torch.nn.TripletMarginLoss.html)
# 
# - este o functie mai interesanta , ea are definit intern un forward si backward si returneaza un tensor
# 

# In[19]:


criterion_triplet = nn.TripletMarginLoss(margin=0.5, p=2)


# In[ ]:


def train_epoch_triplet(model, loader, optimizer,scheduler, device, contrastive_weight=CONTRASTIVE_WEIGHT):

    model.train()

    # adaugam scaler
    scaler = GradScaler()

    total_loss = 0.0
    total_ce = 0.0
    total_triplet = 0.0

    for i, batch in enumerate(loader):
        ids_a = batch['input_ids_a'].to(device, non_blocking=True)
        mask_a = batch['attention_mask_a'].to(device, non_blocking=True)

        ids_p = batch['input_ids_p'].to(device, non_blocking=True)
        mask_p = batch['attention_mask_p'].to(device, non_blocking=True)

        ids_n = batch['input_ids_n'].to(device, non_blocking=True)
        mask_n = batch['attention_mask_n'].to(device, non_blocking=True)

        labels = batch['label'].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # si autocast
        with autocast():
            logits, emb_a = model(ids_a, mask_a)
            _, emb_p = model(ids_p, mask_p)
            _, emb_n = model(ids_n, mask_n)

            # cross entropy
            loss_ce = F.cross_entropy(logits, labels)
            # triplet
            loss_triplet = F.triplet_margin_loss(emb_a, emb_p, emb_n, margin=0.5)
            # contrastive
            loss = loss_ce + contrastive_weight * loss_triplet

        # backprop
        # la autoscale facem pe scaler backrpop 
        scaler.scale(loss).backward()
        # loss.backward()

        # clip la gradient
        # cand facem cu scaler :
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # cica e bine sa pui 1.0 aci nu 1

        # optimizer.step() # actualizam paametrii pe baza gradientilor calculati anterior
        scaler.step(optimizer)
        scaler.update()
        scheduler.step() # adjustam lr in functie de sheduler

        total_loss += loss.item()
        total_ce += loss_ce.item()
        total_triplet += loss_triplet.item()

        if i % 100 == 0:
            print(f"  Batch {i}/{len(loader)} "
                  f"loss: {loss.item():.4f} "
                  f"CE: {loss_ce.item():.4f} "
                  f"Triplet: {loss_triplet.item():.4f}")

    n = len(loader)
    return total_loss/n, total_ce/n, total_triplet/n


# ### Pas 7 : configurari Data loadere + run experiment
# - discutie: ce facem cu val? ce facem cu test? le pastram asa in forma actuala adica ca AuthorDataset? 
# - putem sa le lasam asa si sa clasificam cu classifierul direct pentru ca el tine cont de ce a fost antrenat cu projection head 
# 

# In[21]:


import hashlib
from collections import Counter

def short_hash(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:8]


def inspect_triplets_across_epochs(
    texts_train_w,
    y_train_w,
    hard_negatives,
    samples_per_author=5,
    seed=42,
    epochs_to_check=3,
    n_items=20
):
    all_epoch_triplets = {}

    for epoch in range(epochs_to_check):
        print(f"\n=== EPOCH {epoch} ===")

        ds = TripleAuthorDataset(
            texts_train_w,
            y_train_w,
            hard_negatives=hard_negatives,
            samples_per_author=samples_per_author,
            epoch=epoch,
            seed=seed
        )

        triplet_signatures = []

        for idx in range(min(n_items, len(ds))):
            item = ds[idx]

            a = short_hash(item["anchor"])
            p = short_hash(item["positive"])
            n = short_hash(item["negative"])
            label = item["label"]

            signature = (label, a, p, n)
            triplet_signatures.append(signature)

            print(
                f"idx={idx:03d} | "
                f"label={label} | "
                f"A={a} | P={p} | N={n}"
            )

        all_epoch_triplets[epoch] = set(triplet_signatures)

    print("\n=== COMPARAȚIE ÎNTRE EPOCI ===")

    for e1 in range(epochs_to_check):
        for e2 in range(e1 + 1, epochs_to_check):
            common = all_epoch_triplets[e1] & all_epoch_triplets[e2]
            total = len(all_epoch_triplets[e1] | all_epoch_triplets[e2])

            overlap = len(common) / total if total > 0 else 0.0

            print(
                f"Epoch {e1} vs Epoch {e2}: "
                f"{len(common)} triplete comune | overlap={overlap:.2%}"
            )


# In[22]:


def validate_triplet_dataset(
    texts_train_w,
    y_train_w,
    hard_negatives,
    samples_per_author=20,
    epoch=0,
    seed=42,
    n_items=200
):
    ds = TripleAuthorDataset(
        texts_train_w,
        y_train_w,
        hard_negatives=hard_negatives,
        samples_per_author=samples_per_author,
        epoch=epoch,
        seed=seed
    )

    text_to_labels = {}
    for text, label in zip(texts_train_w, y_train_w):
        text_to_labels.setdefault(text, set()).add(label)

    errors = 0
    author_counts = Counter()
    neg_author_counts = Counter()

    for idx in range(min(n_items, len(ds))):
        item = ds[idx]

        anchor = item["anchor"]
        positive = item["positive"]
        negative = item["negative"]
        label = item["label"]

        anchor_labels = text_to_labels.get(anchor, set())
        positive_labels = text_to_labels.get(positive, set())
        negative_labels = text_to_labels.get(negative, set())

        author_counts[label] += 1

        if label not in anchor_labels:
            print(f"ERROR idx={idx}: anchor nu are label-ul corect")
            errors += 1

        if label not in positive_labels:
            print(f"ERROR idx={idx}: positive nu are label-ul corect")
            errors += 1

        if label in negative_labels:
            print(f"ERROR idx={idx}: negative are același label ca anchor")
            errors += 1

        if anchor == positive:
            print(f"ERROR idx={idx}: anchor == positive")
            errors += 1

        for neg_label in negative_labels:
            neg_author_counts[neg_label] += 1

    print("\n=== VALIDARE TRIPLETE ===")
    print(f"Triplete verificate: {min(n_items, len(ds))}")
    print(f"Erori: {errors}")

    print("\nDistribuție anchor authors:")
    print(author_counts)

    print("\nDistribuție negative authors:")
    print(neg_author_counts)

    if errors == 0:
        print("OK: tripletele respectă regulile de bază.")


# In[23]:


def run_triplet_experiment(
        texts_train_w,
        y_train_w,
        val_loader,
        label_encoder,
        num_labels,
        device,
        dataset_name="",
        epochs=6,
        samples_per_author=50,
        batch_size=8,
        contrastive_weight=0.3,
        top_k=5,
        seed=42):

    set_seed(seed)


    # =====================================================
    # HARD NEGATIVE MINING
    # =====================================================
    sentence_model = SentenceTransformer(ST_MODEL, device=str(device))

    print("\n==========CALCULARE PROFILE AUTORI==============")

    author_profiles = compute_author_profiles(sentence_model,texts_train_w, y_train_w, label_encoder, samples_per_author)

    print("\n==========CALCULARE HARD NEGATIVE==============")

    hard_negatives = compute_hard_negatives(author_profiles, label_encoder, top_k=top_k)

    # eliberam memoria modelului pt propozitii
    del sentence_model
    torch.cuda.empty_cache()

    inspect_triplets_across_epochs(
    texts_train_w,
    y_train_w,
    hard_negatives,
    samples_per_author=5,
    seed=42,
    epochs_to_check=3,
    n_items=30
)
    validate_triplet_dataset(
    texts_train_w,
    y_train_w,
    hard_negatives,
    samples_per_author=20,
    epoch=0,
    seed=42,
    n_items=300
)

    # ============= BERT BACKBONE =============

    model = TripleBERTAuthorship(num_authors=num_labels).to(device)

    # Dataloader temporar pt scheduler

    temp_loader = make_triplet_loader(
        texts_train_w,
        y_train_w,
        hard_negatives,
        batch_size=batch_size,
        samples_per_author=samples_per_author,
        epoch=0,
        seed=seed
    )

    steps_per_epoch = len(temp_loader)

    del temp_loader

    # Optimizer si scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=steps_per_epoch // 10, num_training_steps=steps_per_epoch*epochs)

    # =====================================================
    # OUTPUT
    # =====================================================

    best_f1 = 0

    dataset_output_dir = (
        os.path.join(OUTPUT_DIR, dataset_name)
        if dataset_name else OUTPUT_DIR
    )

    os.makedirs(dataset_output_dir, exist_ok=True)

    model_path = os.path.join(
        dataset_output_dir,
        f"best_triplet_{dataset_name}.pt"
    )

    # =====================================================
    # ISTORIC
    # =====================================================

    train_losses = []
    val_losses = []

    train_f1s = []
    val_f1s = []

    train_accs = []
    val_accs = []

    train_ces = []
    train_triplets = []

    # =====================================================
    # TRAIN LOOP
    # =====================================================

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        # recreem datasetul la fiecare epoca
        # pentru sampling diferit dar reproductibil

        train_loader = make_triplet_loader(
            texts_train_w,
            y_train_w,
            hard_negatives,
            batch_size=batch_size,
            samples_per_author=samples_per_author,
            epoch=epoch,
            seed=seed
        )

        # train
        train_loss, ce, triplet = train_epoch_triplet(model, train_loader, optimizer, scheduler, device, contrastive_weight)

        # evaluate train
        _, train_f1, train_acc = eval_epoch_triplet(
            model,
            train_loader,
            device
        )

        val_loss, val_f1, val_acc = eval_epoch_contrastive_val(
            model,
            val_loader,
            device
        )

        # salvam 
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        train_f1s.append(train_f1)
        val_f1s.append(val_f1)

        train_accs.append(train_acc)
        val_accs.append(val_acc)

        train_ces.append(ce)
        train_triplets.append(triplet)

        print(
            f"Train - "
            f"loss: {train_loss:.4f} "
            f"CE: {ce:.4f} "
            f"Triplet: {triplet:.4f} "
            f"F1: {train_f1:.4f}"
        )

        print(
            f"Val  - "
            f"loss: {val_loss:.4f} "
            f"F1: {val_f1:.4f} "
            f"Acc: {val_acc:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), model_path)
            print(f"  Salvat (F1={val_f1:.4f})")

    print(f"\nBest Val F1: {best_f1:.4f}")
    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )

    # de adaugat grafice!
    plot_training_curves(
        train_losses,
        val_losses,
        train_f1s,
        val_f1s,
        train_accs,
        val_accs,
        dataset_name,
        dataset_output_dir
    )

    plot_contrastive_losses(
        train_ces,
        train_triplets,
        contrastive_weight,
        dataset_name,
        dataset_output_dir
    )

    return model, model_path, {

        'train_losses': train_losses,
        'val_losses': val_losses,

        'train_f1s': train_f1s,
        'val_f1s': val_f1s,

        'train_accs': train_accs,
        'val_accs': val_accs,

        'train_ces': train_ces,
        'train_triplets': train_triplets,

        'best_val_f1': best_f1
    }


# ### Pas 8: implementare functii de evaluare:

# evaluare pe validare:

# In[24]:


def eval_epoch_contrastive_val(model, loader, device):
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                logits, _ = model(input_ids, attention_mask)
                loss = F.cross_entropy(logits, labels)

            preds = torch.argmax(logits, dim=1)

            total_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    f1 = f1_score(all_labels, all_preds, average="macro")
    acc = accuracy_score(all_labels, all_preds)

    return avg_loss, f1, acc


# In[25]:


def eval_epoch_triplet(model, loader, device):

    """
    Evaluare pentru train loader-ul triplet.

    Loader-ul contine:
        input_ids_a
        attention_mask_a
        label

    Folosim doar anchor-ul pentru clasificare.
    """

    model.eval()
    criterion_ce = nn.CrossEntropyLoss()

    total_loss = 0.0

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for batch in loader:
            # anchor inputs
            ids_a = batch['input_ids_a'].to(device)
            mask_a = batch['attention_mask_a'].to(device)
            labels = batch['label'].to(device)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                # forward
                logits, _ = model(ids_a, mask_a)

                # classification loss
                loss = criterion_ce(logits, labels)

            # predictii
            preds = torch.argmax(
                logits,
                dim=-1
            )

            # acumulare
            total_loss += loss.item()

            all_preds.extend(
                preds.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    # metrici finale
    f1 = f1_score(
        all_labels,
        all_preds,
        average='macro'
    )

    acc = accuracy_score(
        all_labels,
        all_preds
    )

    return (
        total_loss / len(loader),
        f1,
        acc
    )


# si functia care evaluaeaza folosind clasificatorul 

# In[26]:


def eval_epoch_classifier(model, loader, device):

    """
    Evaluare pentru validation/test loader normal.

    Loader-ul contine:
        input_ids
        attention_mask
        label
    """

    model.eval()

    criterion_ce = nn.CrossEntropyLoss()

    total_loss = 0.0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            # input normal
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                # forward
                logits, _ = model(input_ids, attention_mask)
                # loss
                loss = criterion_ce(logits,labels)

            # predictii
            preds = torch.argmax(logits,dim=-1)

            # acumulare
            total_loss += loss.item()

            all_preds.extend(preds.cpu().numpy())

            all_labels.extend(labels.cpu().numpy())

    # metrici
    f1 = f1_score(
        all_labels,
        all_preds,
        average='macro'
    )

    acc = accuracy_score(
        all_labels,
        all_preds
    )

    return (
        total_loss / len(loader),
        f1,
        acc
    )


# ### Pas 9: echivalentul main , unde pregatim ferestrele si dam start antrenarii

# In[27]:


print(f"CUDA available safe check:")

try:
    cuda_ok = torch.cuda.is_available()
    print(cuda_ok)

    if cuda_ok:
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
except RuntimeError as e:
    print("CUDA error:", e)
    cuda_ok = False

device = torch.device("cuda" if cuda_ok else "cpu")
print(f"Device: {device}")


# Avem nevoie de o functie care construieste user embeddings pentru evaluarea folosind projection head  <br>
# document
#    -> ferestre
#       -> embeddings ferestre
#           -> media = embedding document
#                -> media peste documente = embedding autor

# In[28]:


def build_user_embeddings(model, texts_train, y_train,
                          tokenizer, device, le,
                          max_length=MAX_LEN, batch_size=32):
    model.eval()
    author_embeddings = {}

    with torch.no_grad():
        for author_idx, author_name in enumerate(le.classes_):
            indices = [i for i, y in enumerate(y_train) if y == author_idx]
            author_texts = [texts_train[i] for i in indices]

            if len(author_texts) == 0:
                continue

            embs = []

            for start in range(0, len(author_texts), batch_size):
                batch_texts = author_texts[start:start + batch_size]

                encoded = tokenizer(
                    batch_texts,
                    max_length=max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt"
                )

                emb = model.get_embedding(
                    encoded["input_ids"].to(device),
                    encoded["attention_mask"].to(device)
                )

                embs.append(emb.cpu())

            all_embs = torch.cat(embs, dim=0)

            author_emb = torch.mean(all_embs, dim=0)
            author_emb = F.normalize(author_emb, p=2, dim=0)

            author_embeddings[author_name] = author_emb

            print(f"  {author_name}: {len(indices)} ferestre")

    return author_embeddings


# Evaluatea se face din doua perspective: cu clasificator direct pe text sau folosind user embeddings construite in build emmbeddings

# In[29]:


def predict_with_user_embeddings(model, text, author_embeddings,
                                 tokenizer, le, device,
                                 max_length=MAX_LEN, overlap=OVERLAP):
    model.eval()

    # 1. impartim textul în ferestre, exact ca la train
    windows = split_into_windows(
        text,
        tokenizer,
        max_length=max_length,
        overlap=overlap
    )

    window_embs = []

    with torch.no_grad():
        for window_text in windows:
            # 2. Tokenize fiecare fereastra
            encoded = tokenizer(
                window_text,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )

            # 3. Scoatem embeddingul ferestrei
            emb = model.get_embedding(
                encoded["input_ids"].to(device),
                encoded["attention_mask"].to(device)
            )

            window_embs.append(emb.cpu())

    # 4. Facem embeddingul documentului = media embeddingurilor ferestrelor
    all_window_embs = torch.cat(window_embs, dim=0)

    doc_emb = torch.mean(all_window_embs, dim=0)
    doc_emb = F.normalize(doc_emb, p=2, dim=0)

    # 5. Comparam documentul cu embeddingul fiecarui autor
    similarities = {
        author_name: F.cosine_similarity(
            doc_emb.unsqueeze(0),
            author_emb.unsqueeze(0)
        ).item()
        for author_name, author_emb in author_embeddings.items()
    }

    # 6. Alegem autorul cel mai apropiat
    pred_author = max(similarities, key=similarities.get)

    # 7. Transformam numele autorului în label numeric
    return le.transform([pred_author])[0]


# si varianta simpla de clasificator

# In[30]:


def predict_with_voting_contrastive(
    model,
    text,
    tokenizer,
    device,
    max_length=MAX_LEN,
    overlap=OVERLAP
):
    # window -> logits -> autor
    model.eval()

    windows = split_into_windows(
        text,
        tokenizer,
        max_length=max_length,
        overlap=overlap
    )

    predictions = []

    with torch.no_grad():
        for window_text in windows:
            encoded = tokenizer(
                window_text,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )

            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            logits, _ = model(input_ids, attention_mask)
            pred = torch.argmax(logits, dim=-1).item()
            predictions.append(pred)

    # Majority voting
    if predictions:
        final_pred = Counter(predictions).most_common(1)[0][0]
    else:
        final_pred = 0  # fallback

    return final_pred


# In[31]:


def evaluate_contrastive(model, texts_test, y_test,
                         tokenizer, le, device,
                         dataset_name="", model_path=None,
                         author_embeddings=None, out_dir="."):
    """
    Evaluare completă cu ambele variante:
    A. Clasificator + voting
    B. User embeddings + cosine similarity
    """

    # modelul salvat
    if model_path:
        print(f"Încărcare model: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))

    model.to(device)
    model.eval()

    # ========================================================================
    # Varianta A: Clasificator + Voting
    # ========================================================================
    print(f"\n{'='*60}")
    print("Varianta A: Clasificator + Majority Voting")
    print(f"{'='*60}")

    preds_a = []

    for i, text in enumerate(texts_test):
        pred = predict_with_voting_contrastive(
            model, text, tokenizer, device
        )
        preds_a.append(pred)

        if i % 100 == 0:
            print(f"  Progress: {i}/{len(texts_test)}")

    # Metrici
    f1_a = f1_score(y_test, preds_a, average="macro")
    acc_a = accuracy_score(y_test, preds_a)

    print(f"\nRezultate Varianta A:")
    print(f"F1:  {f1_a:.4f}")
    print(f"Acc: {acc_a:.4f}\n")

    print(classification_report(
        y_test,
        preds_a,
        labels=list(range(len(le.classes_))),
        target_names=le.classes_,
        zero_division=0
    ))

    # ========================================================================
    # Varianta B: User Embeddings + Cosine
    # ========================================================================
    f1_b, acc_b, preds_b = None, None, None

    if author_embeddings is not None:
        print(f"\n{'='*60}")
        print("Varianta B: User Embeddings + Cosine Similarity")
        print(f"{'='*60}")

        preds_b = []

        for i, text in enumerate(texts_test):
            pred = predict_with_user_embeddings(
                model, text, author_embeddings,
                tokenizer, le, device
            )
            preds_b.append(pred)

            if i % 100 == 0:
                print(f"  Progress: {i}/{len(texts_test)}")

        # Metrici
        f1_b = f1_score(y_test, preds_b, average="macro")
        acc_b = accuracy_score(y_test, preds_b)

        print(f"\nRezultate Varianta B:")
        print(f"F1:  {f1_b:.4f}")
        print(f"Acc: {acc_b:.4f}\n")

        print(classification_report(
            y_test,
            preds_b,
            labels=list(range(len(le.classes_))),
            target_names=le.classes_,
            zero_division=0
        ))
    else:
        print(f"\n Varianta B SKIP: author_embeddings = None")

    return preds_a, f1_a, acc_a, preds_b, f1_b, acc_b


# In[32]:


def implement_contrastive(train_df, val_df, test_df, dataset_name):
    # dataloader si ferestre

    print("\n=== Pregatire date ===")
    le, texts_train_w, y_train_w, \
    texts_val, y_val, \
    texts_test, y_test, \
    val_loader, test_loader = build_dataloaders(
        train_df, val_df, test_df,
        tokenizer, overlap=OVERLAP, seed=42
    )

    num_labels = len(le.classes_)
    print(f"Numar autori: {num_labels}")

    print("\n=== Antrenare ===")
    model, model_path, history = run_triplet_experiment(
        texts_train_w,
        y_train_w,
        val_loader,
        le,                 # lipsea label_encoder
        num_labels=num_labels,
        device=device,
        dataset_name=dataset_name,
        epochs=6,
        samples_per_author=50,
        batch_size=8,
        contrastive_weight=0.3,
        top_k=5,
        seed=42
    )

    print("\n=== User Embeddings ===")
    user_embs = build_user_embeddings(
        model, texts_train_w, y_train_w,
        tokenizer, device, le
    )

    print("\n=== Evaluare test ===")
    preds_a, f1_a, acc_a, preds_b, f1_b, acc_b = evaluate_contrastive(
        model, texts_test, y_test,
        tokenizer, le, device,
        dataset_name='top20_ro_triplet',
        model_path=model_path,
        author_embeddings=user_embs
    )

    plot_confusion_matrix(
        y_test, preds_a, le, 'top20_ro_triplet_clf', 'test'
    )
    if preds_b is not None:
        plot_confusion_matrix(
            y_test, preds_b, le, 'top20_ro_triplet_emb', 'test'
        )

    results = {
        "dataset": dataset_name,
        "f1_clf": f1_a,
        "acc_clf": acc_a,
        "f1_emb": f1_b,
        "acc_emb": acc_b,
        "best_val_f1": history["best_val_f1"]
    }

    with open(f"{OUTPUT_DIR}/results_triplet_{dataset_name}.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nVarianta A: F1={f1_a:.4f} | Acc={acc_a:.4f}")

    if f1_b is not None:
        print(f"Varianta B: F1={f1_b:.4f} | Acc={acc_b:.4f}")

    return {
        "model": model,
        "model_path": model_path,
        "history": history,
        "user_embs": user_embs,
        "preds_a": preds_a,
        "f1_a": f1_a,
        "acc_a": acc_a,
        "preds_b": preds_b,
        "f1_b": f1_b,
        "acc_b": acc_b
    }



# ### Rulare pe datasets

# In[33]:


train_df = pd.read_csv("final_stilo/top20_ro_train.csv")
val_df   = pd.read_csv("final_stilo/top20_ro_val.csv")
test_df  = pd.read_csv("final_stilo/top20_ro_test.csv")


# In[ ]:


results = implement_contrastive(
    train_df,
    val_df,
    test_df,
    "top20_ro"
)


# In[ ]:




