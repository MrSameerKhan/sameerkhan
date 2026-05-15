"""
Session 7 — BiLSTM for NER
data.py: Vocab building, train/val/test split, NERDataset, padded DataLoader.

Pipeline:
    get_ner_data(n_total=200)  → 200 (tokens, tags) examples
    split_data(seed=42)        → 160 train / 20 val / 20 test
    build_vocab(train_only)    → {<PAD>:0, <UNK>:1, "Sarah":2, ...}
    build_label_vocab(train)   → {<PAD>:0, "O":1, "B-PERSON":2, ...}
    NERDataset + collate_fn    → padded batches with lengths
"""

import sys
import os
import random
import torch
from torch.utils.data import Dataset, DataLoader

_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_DIR, "..", "..", "..")
sys.path.insert(0, _ROOT)

from code_practice.shared_dataset import get_ner_data

PAD_TOKEN    = "<PAD>"
UNK_TOKEN    = "<UNK>"
PAD_TOKEN_ID = 0
UNK_TOKEN_ID = 1
PAD_LABEL_ID = 0   # used in CrossEntropyLoss(ignore_index=PAD_LABEL_ID)


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------

def split_data(data, seed=42, train_ratio=0.8, val_ratio=0.1):
    rng = random.Random(seed)
    data = list(data)
    rng.shuffle(data)
    n       = len(data)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    return data[:n_train], data[n_train:n_train + n_val], data[n_train + n_val:]


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

def build_vocab(train_data):
    """Build word vocab from train set only — no data leakage."""
    word2idx = {PAD_TOKEN: PAD_TOKEN_ID, UNK_TOKEN: UNK_TOKEN_ID}
    for tokens, _ in train_data:
        for tok in tokens:
            if tok not in word2idx:
                word2idx[tok] = len(word2idx)
    idx2word = {v: k for k, v in word2idx.items()}
    return word2idx, idx2word


def build_label_vocab(train_data):
    """Build label vocab from train set only. PAD label is always index 0."""
    label2idx = {PAD_TOKEN: PAD_LABEL_ID}
    for _, tags in train_data:
        for tag in tags:
            if tag not in label2idx:
                label2idx[tag] = len(label2idx)
    idx2label = {v: k for k, v in label2idx.items()}
    return label2idx, idx2label


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class NERDataset(Dataset):
    def __init__(self, data, word2idx, label2idx):
        self.data      = data
        self.word2idx  = word2idx
        self.label2idx = label2idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens, tags = self.data[idx]
        token_ids = [self.word2idx.get(t, UNK_TOKEN_ID) for t in tokens]
        label_ids = [self.label2idx[t] for t in tags]
        return (torch.tensor(token_ids, dtype=torch.long),
                torch.tensor(label_ids, dtype=torch.long))


# ---------------------------------------------------------------------------
# Collate — dynamic padding per batch
# ---------------------------------------------------------------------------

def collate_fn(batch):
    """
    For each batch:
    1. Find max length in this batch
    2. Pad shorter token sequences with PAD_TOKEN_ID (0)
    3. Pad shorter label sequences with PAD_LABEL_ID (0)
    4. Return (tokens, labels, lengths)
    """
    token_seqs, label_seqs = zip(*batch)
    lengths    = torch.tensor([len(t) for t in token_seqs])
    max_len    = int(lengths.max().item())

    token_padded = torch.full((len(batch), max_len), PAD_TOKEN_ID, dtype=torch.long)
    label_padded = torch.full((len(batch), max_len), PAD_LABEL_ID, dtype=torch.long)

    for i, (t, l) in enumerate(zip(token_seqs, label_seqs)):
        token_padded[i, :len(t)] = t
        label_padded[i, :len(l)] = l

    return token_padded, label_padded, lengths


# ---------------------------------------------------------------------------
# Convenience loader
# ---------------------------------------------------------------------------

def get_loaders(n_total=200, batch_size=32, seed=42):
    """End-to-end: load data → split → build vocab → return DataLoaders."""
    data = get_ner_data(n_total=n_total, seed=seed)
    train_data, val_data, test_data = split_data(data, seed=seed)

    word2idx, idx2word   = build_vocab(train_data)
    label2idx, idx2label = build_label_vocab(train_data)

    train_ds = NERDataset(train_data, word2idx, label2idx)
    val_ds   = NERDataset(val_data,   word2idx, label2idx)
    test_ds  = NERDataset(test_data,  word2idx, label2idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return (
        (train_loader, val_loader, test_loader),
        word2idx, idx2word, label2idx, idx2label,
        train_data, val_data, test_data,
    )
