"""
Session 8 — Bahdanau Attention on BiLSTM
data.py: Tokenize text, vocab building, classification dataset, padded batches with mask.

Pipeline:
    get_classification_data(n_total=200)  → 200 (text, label) pairs
    split_data(seed=42, 80/10/10)         → 160 train / 20 val / 20 test
    build_vocab(train_only)               → {<PAD>:0, <UNK>:1, ...}
    build_label_vocab(train_only)         → {label: idx}
    ClassificationDataset + collate_fn    → padded batches with attention mask
"""

import sys
import os
import re
import random
import torch
from torch.utils.data import Dataset, DataLoader

_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_DIR, "..", "..", "..")
sys.path.insert(0, _ROOT)

from code_practice.shared_dataset import get_classification_data

PAD_TOKEN    = "<PAD>"
UNK_TOKEN    = "<UNK>"
PAD_TOKEN_ID = 0
UNK_TOKEN_ID = 1


def tokenize(text: str) -> list:
    """Simple whitespace tokenizer — periods become separate tokens."""
    return text.replace(".", " .").split()


def split_data(data, seed=42, train_ratio=0.8, val_ratio=0.1):
    rng = random.Random(seed)
    data = list(data)
    rng.shuffle(data)
    n       = len(data)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    return data[:n_train], data[n_train:n_train + n_val], data[n_train + n_val:]


def build_vocab(train_data):
    """Word vocab from train set only."""
    word2idx = {PAD_TOKEN: PAD_TOKEN_ID, UNK_TOKEN: UNK_TOKEN_ID}
    for text, _ in train_data:
        for tok in tokenize(text):
            if tok not in word2idx:
                word2idx[tok] = len(word2idx)
    idx2word = {v: k for k, v in word2idx.items()}
    return word2idx, idx2word


def build_label_vocab(train_data):
    """Label vocab from train set only."""
    label2idx = {}
    for _, label in train_data:
        if label not in label2idx:
            label2idx[label] = len(label2idx)
    idx2label = {v: k for k, v in label2idx.items()}
    return label2idx, idx2label


class ClassificationDataset(Dataset):
    def __init__(self, data, word2idx, label2idx):
        self.data      = data
        self.word2idx  = word2idx
        self.label2idx = label2idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        tokens    = tokenize(text)
        token_ids = [self.word2idx.get(t, UNK_TOKEN_ID) for t in tokens]
        label_id  = self.label2idx[label]
        return (torch.tensor(token_ids, dtype=torch.long),
                torch.tensor(label_id,  dtype=torch.long))


def collate_fn(batch):
    """
    Dynamic padding + attention mask.
    Returns (tokens [B,T], labels [B], lengths [B], mask [B,T]).
    mask=1 for real tokens, 0 for padding.
    """
    token_seqs, label_ids = zip(*batch)
    lengths = torch.tensor([len(t) for t in token_seqs])
    max_len = int(lengths.max().item())

    tokens_padded = torch.full((len(batch), max_len), PAD_TOKEN_ID, dtype=torch.long)
    mask          = torch.zeros(len(batch), max_len, dtype=torch.long)

    for i, t in enumerate(token_seqs):
        tokens_padded[i, :len(t)] = t
        mask[i, :len(t)] = 1

    labels = torch.stack(list(label_ids))
    return tokens_padded, labels, lengths, mask


def get_loaders(n_total=200, batch_size=32, seed=42):
    data = get_classification_data(n_total=n_total, seed=seed)
    train_data, val_data, test_data = split_data(data, seed=seed)

    word2idx, idx2word   = build_vocab(train_data)
    label2idx, idx2label = build_label_vocab(train_data)

    train_ds = ClassificationDataset(train_data, word2idx, label2idx)
    val_ds   = ClassificationDataset(val_data,   word2idx, label2idx)
    test_ds  = ClassificationDataset(test_data,  word2idx, label2idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return (
        (train_loader, val_loader, test_loader),
        word2idx, idx2word, label2idx, idx2label,
        train_data, val_data, test_data,
    )
