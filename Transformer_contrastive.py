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
from sklearn.metrics import f1_score, accuracy_score
import torch
import numpy as np
import random

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter
import unicodedata
from torch.utils.data import Sampler
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import json


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("dumitrescustefan/bert-base-romanian-cased-v1")

OUTPUT_DIR = "outputs"
MODEL_PATH = "best_model_contrastive.pt"

EPOCHS = 15
LR = 2e-5
MAX_LEN = 512 # pana acum am testat cu 512 dar e dezastruos pe top 40 moldova , 512 e cel mai bun lasa asa 
CONTRASTIVE_WEIGHT = 0.7
OVERLAP = 0.3 # 0.1 e cam slabut as zice sper sa mearga un pic mai mare

os.makedirs(OUTPUT_DIR, exist_ok=True)

# FUNCTII UTILE le definim sus
def drop_columns(df, columns_to_drop):
    return df.drop(columns=columns_to_drop)

# avem o problema la moldova
# diacriticele aparent nu sunt unicode?
def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFC", text)

    # remove combining chars, deci t de la tap nu va mai avea caciulita sepaarata va fi un singur char
    text = ''.join(c for c in text if not unicodedata.combining(c))

    text = text.replace("„", '"').replace("”", '"')
    text = text.replace("’", "'")

    # remove control chars
    text = ''.join(c for c in text if unicodedata.category(c)[0] != 'C')

    text = text.replace("\n", " ")
    text = text.replace('""', '"')
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def count_unk(texts, tokenizer, n=100):
    total_unk = 0
    total_tokens = 0

    for t in texts[:n]:
        enc = tokenizer(
            t,
            truncation=True,
            max_length=MAX_LEN
        )

        input_ids = enc["input_ids"]
        total_tokens += len(input_ids)

        unk_id = tokenizer.unk_token_id
        total_unk += sum(1 for x in input_ids if x == unk_id)

    return total_unk / total_tokens


def count_unk_absolute(texts, tokenizer, n=100):
    total_unk = 0
    for t in texts[:n]:
        enc = tokenizer(t, truncation=True, max_length=MAX_LEN)
        unk_id = tokenizer.unk_token_id
        total_unk += sum(1 for x in enc["input_ids"] if x == unk_id)
    return total_unk


def count_combining(texts, n=1000):
    count = 0
    for t in texts[:n]:
        if any(unicodedata.combining(c) for c in t):
            count += 1
    return count


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

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
    


# definim o functie in care mapam fiecare autor la ferestre de 512 tokeni
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
        return [text]
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


"""
initial am asa: texts = ["...", "...", "..."]
labels = [0, 0, 1, 2, 1, ...]

1 text = 1 exemplu
eu vreau: (text1, text2, label) = exemplu

Pas1: grupam pe autori Autor 0 = [win1, win2, ..]
"""
class ContrastiveAuthorDataset(Dataset):

  def __init__(self, texts, labels, batches_per_epoch=None, epoch=0, seed=42):

    # ne asiguram ca avem acelasi shuffle pe epoca
    rng = random.Random(seed + epoch)

    # grupam ferestrele pe autori
    # ne vom folosi de un dictionar pt asta
    # author_texts = { autor: [lista texte] }
    author_texts = {}
    for text, label in zip(texts, labels):
      # daca nu il am deja in dex il instantiez
      if label not in author_texts:
        author_texts[label] = []
      author_texts[label].append(text)

    #ok dar daca nu am macar 2 ferestre la un autor nu il pot pune pe ala
    self.valid_authors = sorted([
        a for a, t in author_texts.items() if len(t) >= 2
    ])

    # ok acum putem sa cream perechile pe autor
    # 1. copiezi lista de texte
    # 2. faci shuffle
    # 3. creezi perechi consecutive:
    #   (t1,t2), (t3,t4)
    # salvam ca:
    # author_pairs = {autor: [(t1,t2), (t3,t4)}

    # dar trebuie calculat minimul de ferestre
    min_windows = min(
        len(author_texts[a]) for a in self.valid_authors
    )

    # putem configura doar un anumit nr de perechi per autor
    # conditionat de numarul minim de ferestre
    # pe care il impartim la 2 ca luam perechi
    max_pairs = min_windows // 2

    if batches_per_epoch is None:
      n_pairs = max_pairs
    else:
      n_pairs = min(batches_per_epoch, max_pairs)

    # nu stiu cat de relevanta e asta sincer:
    # n_pairs = min(batches_per_epoch, max_perechi)
    print(f"Autori: {len(self.valid_authors)}")
    print(f"Min ferestre per autor: {min_windows}")
    print(f"Perechi per autor: {max_pairs}")

    # ok acum ca stim max_pairs trebuie sa generam fix atatea per autor
    # pentru fiecare index k :
    #   batch_k = []
    # pt fiecare autor:
    # iau perechea k  si adaug in batch
    self.pairs = []       #(text1, text2)
    self.pair_labels = [] # (label)

    for author in self.valid_authors:
      windows = author_texts[author].copy()
      rng.shuffle(windows) # riksyy
      # acum iau perechile consecutive dupa shuffle
      for i in range(0, n_pairs * 2, 2): # daca e impar min_windows??
        self.pairs.append((windows[i], windows[i+1]))
        self.pair_labels.append(author)

    print(f"Total perechi: {len(self.pairs)}")


  def __len__(self):
    return len(self.pairs)


  def __getitem__(self, index):
    # scoatem de la index exemplul
    text1, text2 = self.pairs[index]
    label = self.pair_labels[index]

    return {
        'text1': text1,
        'text2': text2,
        'label': label
    }
  


class UniqueAuthorBatchSampler(Sampler):
  def __init__(self, dataset, seed=42):
    # grupam indici pe autor
    #   author_indices = {}

    # pentru fiecare index i:
    #   label = pair_labels[i]
    #   author_indices[label].append(i)

    # obtinem: A = [0,1], B = [2,4], C = [3,5]
    self.seed = seed

    self.author_indices = {}
    for idx, label in enumerate(dataset.pair_labels):
      if label not in self.author_indices:
        self.author_indices[label] = []
      self.author_indices[label].append(idx)

    # ok cate batch-uri putem avea?
    # as zice nr minim de exemple autor
    self.authors = sorted(self.author_indices.keys())
    self.batch_size = len(self.authors)

    self.n_batches = min(
      len(idxs) for idxs in self.author_indices.values()
    )
    print(f"Batch size: {self.batch_size}")
    print(f"Batches per epoca: {self.n_batches}")

  def __iter__(self):
    rng = random.Random(self.seed)

    # shuffle indicii per autor
    # pentru fiecare autor:
    #   amesteci lista de indici (pozitiile unde apare un autor)
    shuffled = {
        a: rng.sample(idxs, len(idxs))
        for a, idxs in self.author_indices.items()
    }
    # tranaformam indicii in iteratori
    iters = {a: iter(idxs) for a, idxs in shuffled.items()}
    for _ in range(self.n_batches):
      batch = []
      # acum shuffle oridne autori  in batch
      authors_shuffled = self.authors.copy()
      rng.shuffle(authors_shuffled)

      for author in authors_shuffled:
        batch.append(next(iters[author]))

      yield batch


  def __len__(self):
    return self.n_batches


def build_contrastive_dataloaders(train_df, val_df, test_df,
                                   tokenizer,
                                   batches_per_epoch=None,
                                   seed=42):
    le = LabelEncoder()
    le.fit(train_df['author'])

    y_train = le.transform(train_df['author'])
    y_val   = le.transform(val_df['author'])
    y_test  = le.transform(test_df['author'])

    texts_train = train_df['text'].apply(clean_text).tolist()
    texts_val   = val_df['text'].apply(clean_text).tolist()
    texts_test  = test_df['text'].apply(clean_text).tolist()

    # ferestre
    texts_train_w, y_train_w = prepare_texts_with_windows(
        texts_train, y_train, tokenizer
    )
    # texts_val_w, y_val_w = prepare_texts_with_windows(
    #     texts_val, y_val, tokenizer
    # )
    # val loader normal
    val_ds = AuthorDataset(texts_val, y_val, tokenizer)
    val_loader = DataLoader(
        val_ds, batch_size=8,
        shuffle=False, num_workers=0, pin_memory=True
    )
    # test loader normal
    test_ds     = AuthorDataset(texts_test, y_test, tokenizer)
    test_loader = DataLoader(
        test_ds, batch_size=8,
        shuffle=False, num_workers=0, pin_memory=True
    )

    return (le,
            texts_train_w, y_train_w,
            texts_val, y_val, val_loader,
            texts_test,    y_test,
            test_loader)


def contrastive_collate_fn(batch):
  """
  Avem un batch de forma:
  [
  {"text1": "...", "text2": "...", "label": 0},
  {"text1": "...", "text2": "...", "label": 1},
  ...
  ]

  Cum il facem tensor?
  Trebuie sa extragem textele le bagam in lista si le tokenizam = input_ids + attention_mask (adica le face numere :)
  Asa acum trebuie sa cream view1 si view2
  1. ia batch (listă de dicționare)
 2. extrage:
   - texts1
   - texts2
   - labels
 3. tokenizează texts1
 4. tokenizează texts2
 5. convertește labels în tensor
 6. returnează totul organizat
  """
  # extragem
  #batch = batch[0]
  texts1 = [item['text1'] for item in batch]
  texts2 = [item['text2'] for item in batch]
  labels = [item['label'] for item in batch]

  enc1 = tokenizer(
      texts1, max_length=MAX_LEN,
      truncation=True, padding='max_length',
      return_tensors='pt'
  )

  enc2 =  tokenizer(
      texts2, max_length=MAX_LEN,
      truncation=True, padding='max_length',
      return_tensors='pt'
  )

  return {
      'input_ids1': enc1['input_ids'],
      'attention_mask_1': enc1['attention_mask'],
      'input_ids2': enc2['input_ids'],
      'attention_mask_2': enc2['attention_mask'],
      'label': torch.tensor(labels, dtype=torch.long) # se supara CE daca nu e long
  }


def make_loaders_for_epoch(texts_train_w, y_train_w, batches_per_epoch, epoch, seed):
    train_ds = ContrastiveAuthorDataset(
        texts_train_w, y_train_w,
        batches_per_epoch=batches_per_epoch,
        epoch=epoch, seed=seed
    )

    train_sampler = UniqueAuthorBatchSampler(train_ds, seed=seed+epoch)

    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        collate_fn=contrastive_collate_fn,
        num_workers=0, pin_memory=True
    )

    return train_loader


def info_nce(z1, z2, temperature=0.07):
  # z1, z2 (N,d)  normalizate
  # similaritate (N,N)
  logits = z1 @ z2.T / temperature
  targets = torch.arange(z1.size(0), device=z1.device)

  loss_12 = F.cross_entropy(logits, targets)
  loss_21 = F.cross_entropy(logits.T, targets)

  return 0.5 * (loss_12 + loss_21), logits


# si loss total ce + lambda * info_nce
def compute_loss(logits, labels, embs_1, embs_2,
                 lambda_contrastive=0.3, temperature=0.07):
    loss_ce  = F.cross_entropy(logits, labels)
    loss_nce, _ = info_nce(embs_1, embs_2, temperature)
    loss     = loss_ce + lambda_contrastive * loss_nce
    return loss, loss_ce, loss_nce


# facem mean pooling peste predictii
# media peste tokeni
# def mean_pooling(bert_output, attention_mask):
#   # scoatem embedding
#   token_embeddings = bert_output.last_hidden_state

#   # [1,1,1,0,0] - [[1],[1],[1],[0],[0]]
#   # ca sa il pot inmulti cu embeddings
#   # (N, seq_len) - (N, seq_len, 1)
#   mask_expanded = attention_mask.unsqueeze(-1).float()
#   sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim = 1)
#   sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
#   return sum_embeddings / sum_mask


class ContrastiveBERTAuthorship(nn.Module):
    def __init__(self, num_authors, projection_dim=128, dropout=0.2):
        super().__init__()

        self.bert = AutoModel.from_pretrained(
            "dumitrescustefan/bert-base-romanian-cased-v1"
        )

        # explicam aici de ce: nu mai salveaza activarile
        # recalculeaza la baxward
        # salvam memorie nu si timp
        # self.bert.gradient_checkpointing_enable()
        # totusi nu e bine aiic ca va avea inferenta mai lenta
        # il bagam in run contrastive
        self.dropout = nn.Dropout(dropout)

        # clasificare directa
        self.classifier = nn.Linear(768, num_authors)

        # pentru n-pair loss si user embeddings
        self.projection_head = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, projection_dim)
        )

    def forward(self, input_ids, attention_mask):
        bert_out   = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        #pooled     = mean_pooling(bert_out, attention_mask)
        # e cam rau cu mean pooling o sa bag cls din nou
        pooled = bert_out.last_hidden_state[:, 0]

        pooled     = self.dropout(pooled)

        logits     = self.classifier(pooled)
        projection = self.projection_head(pooled)
        projection = torch.nn.functional.normalize(projection, p=2, dim=1)

        return logits, projection

    def get_embedding(self, input_ids, attention_mask):
        with torch.no_grad():
            bert_out   = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            #pooled     = mean_pooling(bert_out, attention_mask)
            pooled = bert_out.last_hidden_state[:, 0]
            projection = self.projection_head(pooled)
            projection = torch.nn.functional.normalize(projection, p=2, dim=1)
        return projection


def extract_embeddings(model, loader, device):
    model.eval()
    all_embs = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            _, embs = model(input_ids, attention_mask)

            all_embs.append(embs.cpu())
            all_labels.append(labels.cpu())

    return torch.cat(all_embs), torch.cat(all_labels)


def compute_similarity_stats(embs, labels):
    embs = torch.nn.functional.normalize(embs, dim=1)
    sim_matrix = embs @ embs.T

    labels = labels.numpy()
    sim_matrix = sim_matrix.numpy()

    same = []
    diff = []

    N = len(labels)
    for i in range(N):
        for j in range(i+1, N):
            if labels[i] == labels[j]:
                same.append(sim_matrix[i][j])
            else:
                diff.append(sim_matrix[i][j])

    return same, diff


def save_similarity_plot(same, diff, epoch, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    plt.figure()
    plt.hist(same, bins=50, alpha=0.5, label='same author')
    plt.hist(diff, bins=50, alpha=0.5, label='different author')
    plt.legend()
    plt.title(f"Similarity distribution epoch {epoch}")
    plt.savefig(f"{out_dir}/sim_epoch_{epoch}.png")
    plt.close()


def save_similarity_stats(same, diff, epoch, out_dir):
    import json
    stats = {
        "epoch": epoch,
        "same_mean": float(sum(same)/len(same)),
        "diff_mean": float(sum(diff)/len(diff))
    }

    with open(f"{out_dir}/stats_epoch_{epoch}.json", "w") as f:
        json.dump(stats, f, indent=2)


def train_epoch_contrastive(model, loader, optimizer,
                              scheduler, device,
                              contrastive_weight=0.3, scaler=None):
    
    if scaler is None:
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    model.train()
    criterion_ce = nn.CrossEntropyLoss()
    total_loss   = 0.0
    total_ce     = 0.0
    total_npair  = 0.0

    for i, batch in enumerate(loader):
        input_ids_1      = batch['input_ids1'].to(device, non_blocking=True)
        attention_mask_1 = batch['attention_mask_1'].to(device, non_blocking=True)
        input_ids_2      = batch['input_ids2'].to(device, non_blocking=True)
        attention_mask_2 = batch['attention_mask_2'].to(device, non_blocking=True)
        labels           = batch['label'].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            logits, embs_1 = model(input_ids_1, attention_mask_1)
            _,      embs_2 = model(input_ids_2, attention_mask_2)

            loss_ce    = criterion_ce(logits, labels)
            # loss_nce = info_nce(embs_1, embs_2)[0]  # [0] pentru ca returneaza (loss, logits)
            loss, loss_ce, loss_nce = compute_loss(logits, labels, embs_1, embs_2)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss  += loss.item()
        total_ce    += loss_ce.item()
        total_npair += loss_nce.item()

        if i % 100 == 0:
            print(f"  Batch {i}/{len(loader)} "
                  f"loss: {loss.item():.4f} "
                  f"CE: {loss_ce.item():.4f} "
                  f"Loss infoNCE: {loss_nce.item():.4f}")
            if torch.cuda.is_available():
                print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
                print(f"VRAM folosit: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    n = len(loader)
    return total_loss/n, total_ce/n, total_npair/n


def eval_epoch_contrastive(model, loader, device):
    model.eval()
    criterion_ce = nn.CrossEntropyLoss()
    total_loss   = 0.0
    all_preds    = []
    all_labels   = []

    with torch.no_grad():
        for batch in loader:
            input_ids_1      = batch['input_ids1'].to(device)
            attention_mask_1 = batch['attention_mask_1'].to(device)
            labels           = batch['label'].to(device)

            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                logits, _ = model(input_ids_1, attention_mask_1)
                loss      = criterion_ce(logits, labels)
            
            preds     = torch.argmax(logits, dim=-1)

            total_loss += loss.item()
            all_preds  .extend(preds.cpu().numpy())
            all_labels .extend(labels.cpu().numpy())

    f1  = f1_score(all_labels, all_preds, average='macro')
    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(loader), f1, acc

def eval_epoch_contrastive_val(model, loader, device):
    """Pentru val/test loader normal - are chei input_ids"""
    model.eval()
    criterion_ce = nn.CrossEntropyLoss()
    total_loss   = 0.0
    all_preds    = []
    all_labels   = []

    with torch.no_grad():
        for batch in loader:
            ids    = batch['input_ids'].to(device)
            mask   = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                logits, _ = model(ids, mask)
                loss      = criterion_ce(logits, labels)

            preds = torch.argmax(logits, dim=-1)
            total_loss += loss.item()
            all_preds  .extend(preds.cpu().numpy())
            all_labels .extend(labels.cpu().numpy())

    f1  = f1_score(all_labels, all_preds, average='macro')
    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(loader), f1, acc


def run_contrastive_experiment( texts_train_w, y_train_w,
                                val_loader,
                                num_labels, device,
                                dataset_name="",
                                epochs=6,
                                batches_per_epoch=None,
                                contrastive_weight=0.3,
                                seed=42):
    set_seed(seed)

    model = ContrastiveBERTAuthorship(num_authors=num_labels).to(device)
    model.bert.gradient_checkpointing_enable()  # doar la antrenare

    # estimam steps pentru scheduler
    temp_loader = make_loaders_for_epoch(
        texts_train_w, y_train_w,
        batches_per_epoch, epoch=0, seed=seed
    )
    steps_per_epoch = len(temp_loader)
    del temp_loader

    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=steps_per_epoch // 10,
        num_training_steps=steps_per_epoch * epochs
    )

    best_f1    = 0
    dataset_output_dir = os.path.join(OUTPUT_DIR, dataset_name) if dataset_name else OUTPUT_DIR
    os.makedirs(dataset_output_dir, exist_ok=True)
    model_path = os.path.join(dataset_output_dir, f'best_contrastive_{dataset_name}.pt')

    train_losses, val_losses = [], []
    train_f1s,   val_f1s    = [], []
    train_accs,  val_accs   = [], []
    train_infonce           = []
    # salvam si aceste loss-uri pt grafice separate 
    # ca sa vedem cum influenteaza fiecare loss total
    train_ce_losses = []
    #train_nce_losses = []

    # scaler vezi ca nu merge asa:
    #scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        train_loader = make_loaders_for_epoch(
            texts_train_w, y_train_w,
            batches_per_epoch, epoch=epoch, seed=seed
        )

        train_loss, ce, infonce = train_epoch_contrastive(
            model, train_loader, optimizer, scheduler,
            device, contrastive_weight, scaler=scaler
        )
        _, train_f1, train_acc = eval_epoch_contrastive(
            model, train_loader, device
        )
        val_loss, val_f1, val_acc = eval_epoch_contrastive_val(
            model, val_loader, device
        )

        embs, labels = extract_embeddings(model, val_loader, device)
        same, diff = compute_similarity_stats(embs, labels)

        save_similarity_plot(same, diff, epoch, dataset_output_dir)
        save_similarity_stats(same, diff, epoch, dataset_output_dir)

        train_losses.append(train_loss)
        val_losses  .append(val_loss)
        train_f1s   .append(train_f1)
        val_f1s     .append(val_f1)
        train_accs  .append(train_acc)
        val_accs    .append(val_acc)
        train_infonce.append(infonce)
        train_ce_losses.append(ce)
        #train_nce_losses.append(infonce)

        print(f"Train - loss: {train_loss:.4f} CE: {ce:.4f} "
              f"InfoNCE: {infonce:.4f} F1: {train_f1:.4f}")
        print(f"Val   - loss: {val_loss:.4f} F1: {val_f1:.4f} "
              f"Acc: {val_acc:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), model_path)
            print(f"  Salvat (F1={val_f1:.4f})")

    print(f"\nBest Val F1: {best_f1:.4f}")
    model.load_state_dict(torch.load(model_path, map_location=device))

    model.bert.gradient_checkpointing_disable()

    plot_training_curves(train_losses, val_losses, train_f1s, val_f1s, train_accs, val_accs, dataset_name, dataset_output_dir)

    plot_contrastive_losses(train_ce_losses, train_infonce, contrastive_weight, dataset_name, dataset_output_dir)

    return model, model_path, {
        'train_losses': train_losses, 'val_losses':  val_losses,
        'train_f1s':    train_f1s,   'val_f1s':      val_f1s,
        'train_accs':   train_accs,  'val_accs':     val_accs,
        'train_infonce': train_infonce,'best_val_f1':  best_f1,
        'train_ce_losses': train_ce_losses,
    }


def plot_contrastive_losses(train_ce_losses,
                            train_nce_losses,
                            contrastive_weight=0.3,
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
            f'contrastive_losses_{dataset_name}.png'
        ),
        dpi=150,
        bbox_inches='tight'
    )

    plt.close()


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
    plt.savefig(os.path.join(out_dir, f'training_curves_contrastive_{dataset_name}.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix_bert_fusion(y_true, y_pred, label_encoder,
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
    plt.savefig(os.path.join(out_dir, f'cm_counts_contrastive_{dataset_name}_{split}.png'), dpi=150, bbox_inches='tight')
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
    plt.savefig(os.path.join(out_dir, f'cm_norm_contrastive_{dataset_name}_{split}.png'), dpi=150, bbox_inches='tight')
    plt.close()



def build_user_embeddings(model, texts_train, y_train,
                           tokenizer, device, le,
                           max_length=MAX_LEN, batch_size=32):
    model.eval()
    author_embeddings = {}
    
    for author_idx, author_name in enumerate(le.classes_):
        # gasesti toate textele acestui autor
        indices    = [i for i, y in enumerate(y_train) if y == author_idx]
        autor_texts = [texts_train[i] for i in indices]
        
        embs = []
        for start in range(0, len(autor_texts), batch_size):
            batch_texts = autor_texts[start:start+batch_size]
            encoded = tokenizer(
                batch_texts, max_length=max_length,
                truncation=True, padding='max_length',
                return_tensors='pt'
            )
            emb = model.get_embedding(
                encoded['input_ids'].to(device),
                encoded['attention_mask'].to(device)
            )
            embs.append(emb.cpu())
        
        all_embs = torch.cat(embs, dim=0)
        author_embeddings[author_name] = torch.mean(all_embs, dim=0)
        print(f"  {author_name}: {len(indices)} texte")
    
    return author_embeddings



def predict_with_voting_contrastive(model, text, tokenizer, 
                                     device, max_length=MAX_LEN, overlap=OVERLAP):
    stride = int(max_length * (1 - overlap))
    model.eval()
    
    tokens    = tokenizer(text, truncation=False,
                          add_special_tokens=True, return_tensors='pt')
    input_ids = tokens['input_ids'][0]
    
    predictions = []
    with torch.no_grad():
        for start in range(0, len(input_ids), stride):
            end    = min(start + max_length, len(input_ids))
            window = input_ids[start:end]
            
            if len(window) < max_length:
                padding = torch.zeros(max_length - len(window), dtype=torch.long)
                window  = torch.cat([window, padding])
            
            attn   = (window != 0).long()
            
            # diferenta fata de baseline — model returneaza (logits, projection)
            logits, _ = model(
                window.unsqueeze(0).to(device),
                attn.unsqueeze(0).to(device)
            )
            pred = torch.argmax(logits, dim=-1).item()
            predictions.append(pred)
            
            if end == len(input_ids):
                break
    
    return Counter(predictions).most_common(1)[0][0]


def predict_with_user_embeddings(model, text, author_embeddings,
                                  tokenizer, le, device, max_length=MAX_LEN):
    model.eval()
    encoded = tokenizer(
        text, max_length=max_length,
        truncation=True, padding='max_length',
        return_tensors='pt'
    )
    doc_emb = model.get_embedding(
        encoded['input_ids'].to(device),
        encoded['attention_mask'].to(device)
    ).squeeze(0).cpu()
    
    similarities = {
        author: torch.nn.functional.cosine_similarity(
            doc_emb.unsqueeze(0),
            emb.unsqueeze(0)
        ).item()
        for author, emb in author_embeddings.items()
    }
    
    pred_author = max(similarities, key=similarities.get)
    return le.transform([pred_author])[0]


def evaluate_contrastive(model, texts_test, y_test,
                          tokenizer, le, device,
                          dataset_name="", model_path=None,
                          author_embeddings=None, out_dir="."):
    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=device))
    
    print(f"\n--- Varianta A: clasificator + voting ---")
    preds_a = []
    for i, text in enumerate(texts_test):
        pred = predict_with_voting_contrastive(
            model, text, tokenizer, device
        )
        preds_a.append(pred)
        if i % 100 == 0:
            print(f"  {i}/{len(texts_test)}")
    
    f1_a  = f1_score(y_test, preds_a, average='macro')
    acc_a = accuracy_score(y_test, preds_a)
    print(f"F1: {f1_a:.4f} | Acc: {acc_a:.4f}")
    print(classification_report(y_test, preds_a, target_names=le.classes_))
    
    # varianta B doar daca avem user embeddings
    f1_b, acc_b, preds_b = None, None, None
    if author_embeddings is not None:
        print(f"\n--- Varianta B: user embeddings + cosine ---")
        preds_b = []
        for i, text in enumerate(texts_test):
            pred = predict_with_user_embeddings(
                model, text, author_embeddings,
                tokenizer, le, device
            )
            preds_b.append(pred)
            if i % 100 == 0:
                print(f"  {i}/{len(texts_test)}")
        
        f1_b  = f1_score(y_test, preds_b, average='macro')
        acc_b = accuracy_score(y_test, preds_b)
        print(f"F1: {f1_b:.4f} | Acc: {acc_b:.4f}")
        print(classification_report(y_test, preds_b, target_names=le.classes_))
    
    return preds_a, f1_a, acc_a, preds_b, f1_b, acc_b


def run_dataset(train_path, val_path, test_path, dataset_name):
    print(f"Device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    # -------------------------------------------------------- #
    # CITIRE DATE                                              #
    # -------------------------------------------------------- #
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    # -------------------------------------------------------- #
    # BUILD DATALOADERS                                        #
    # -------------------------------------------------------- #
    print("\n=== Pregatire date ===")
    le, texts_train_w, y_train_w, \
    texts_val, y_val, val_loader, texts_test, y_test, test_loader = build_contrastive_dataloaders(
        train_df, val_df, test_df,
        tokenizer,
        batches_per_epoch=None,
        seed=42
    )
    
    num_labels = len(le.classes_)
    print(f"Numar autori: {num_labels}")

    # -------------------------------------------------------- #
    # ANTRENARE                                                #
    # -------------------------------------------------------- #
    print("\n=== Antrenare ===")
    model, model_path, history = run_contrastive_experiment(
        texts_train_w, y_train_w,
        val_loader,
        num_labels=num_labels,
        device=device,
        dataset_name=dataset_name,
        epochs=EPOCHS,
        batches_per_epoch=None,
        contrastive_weight=CONTRASTIVE_WEIGHT,
        seed=42
    )


    # -------------------------------------------------------- #
    # PLOT TRAINING CURVES                                     #
    # -------------------------------------------------------- #
    plot_training_curves(
        history['train_losses'], history['val_losses'],
        history['train_f1s'],   history['val_f1s'],
        history['train_accs'],  history['val_accs'],
        dataset_name=dataset_name
    )
    
    # -------------------------------------------------------- #
    # USER EMBEDDINGS                                          #
    # -------------------------------------------------------- #
    print("\n=== Construire user embeddings ===")
    user_embs = build_user_embeddings(
        model, texts_train_w, y_train_w,
        tokenizer, device, le
    )

    np.save(f"{OUTPUT_DIR}/user_embeddings_{dataset_name}.npy", user_embs, allow_pickle=True)
    
    # -------------------------------------------------------- #
    # EVALUARE TEST                                            #
    # -------------------------------------------------------- #
    print("\n=== Evaluare test ===")
    preds_a, f1_a, acc_a, preds_b, f1_b, acc_b = evaluate_contrastive(
        model, texts_test, y_test,
        tokenizer, le, device,
        dataset_name=dataset_name,
        model_path=model_path,
        author_embeddings=user_embs
    )

    # -------------------------------------------------------- #
    # CONFUSION MATRIX                                         #
    # -------------------------------------------------------- #
    plot_confusion_matrix_bert_fusion(
        y_test, preds_a, le, dataset_name + '_classifier', 'test'
    )
    if preds_b is not None:
        plot_confusion_matrix_bert_fusion(
            y_test, preds_b, le, dataset_name +'_user_emb', 'test'
        )
    
    # -------------------------------------------------------- #
    # REZULTATE FINALE                                         #
    # -------------------------------------------------------- #
    print("\n=== REZULTATE FINALE ===")
    print(f"Varianta A (clasificator): F1={f1_a:.4f} | Acc={acc_a:.4f}")
    if f1_b:
        print(f"Varianta B (user emb):     F1={f1_b:.4f} | Acc={acc_b:.4f}")
    
    results = {
        'dataset':   dataset_name,
        'f1_clf':    f1_a,
        'acc_clf':   acc_a,
        'f1_emb':    f1_b,
        'acc_emb':   acc_b,
        'best_val_f1': history['best_val_f1']
    }

    with open(f'{OUTPUT_DIR}/results_{dataset_name}.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Rezultate salvate in {OUTPUT_DIR}/results_{dataset_name}.json")
    
def main():

    # run_dataset(
    #     'final_stilo/top20_ro_train.csv',
    #     'final_stilo/top20_ro_val.csv',
    #     'final_stilo/top20_ro_test.csv',
    #     'top20_ro'
    # )

    run_dataset(
        'final_stilo/top20_mo_train.csv',
        'final_stilo/top20_mo_val.csv',
        'final_stilo/top20_mo_test.csv',
        'top20_mo'
    )

    # run_dataset(
    #     'final_stilo/top40_ro_train.csv',
    #     'final_stilo/top40_ro_val.csv',
    #     'final_stilo/top40_ro_test.csv',
    #     'top40_ro'
    # )

    # run_dataset(
    #     'final_stilo/top40_mo_train.csv',
    #     'final_stilo/top40_mo_val.csv',
    #     'final_stilo/top40_mo_test.csv',
    #     'top40_mo'
    # )



if __name__ == '__main__':
    main()