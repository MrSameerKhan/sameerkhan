"""
Phase 2, Session 4 — Positional Encoding
data.py: Same Acme classification task as Sessions 02 and 03, with PyTorch tensor batching.
"""

import sys
import os
import random
import torch

_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_DIR, "..", "..", "..")
sys.path.insert(0, _ROOT)

from code_practice.shared_dataset import get_classification_data

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_IDX   = 0
UNK_IDX   = 1

LABEL2IDX = {"checking": 0, "savings": 1, "loan": 2, "employee": 3, "policy": 4}
IDX2LABEL = {v: k for k, v in LABEL2IDX.items()}


def tokenize(text: str) -> list:
    return (text.lower()
            .replace(".", " .")
            .replace(",", " ,")
            .replace("?", " ?")
            .split())


def build_vocab(train_data: list) -> dict:
    word2idx = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
    for text, _ in train_data:
        for tok in tokenize(text):
            if tok not in word2idx:
                word2idx[tok] = len(word2idx)
    return word2idx


def encode(text: str, word2idx: dict) -> list:
    return [word2idx.get(t, UNK_IDX) for t in tokenize(text)]


def split_data(data, seed=42, train_ratio=0.8, val_ratio=0.1):
    rng  = random.Random(seed)
    data = list(data)
    rng.shuffle(data)
    n       = len(data)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    return data[:n_train], data[n_train:n_train + n_val], data[n_train + n_val:]


def make_batch(examples: list, word2idx: dict, device: torch.device):
    """Pad a list of (text, label) to tensors."""
    ids_list = [encode(text, word2idx) for text, _ in examples]
    labels   = [LABEL2IDX[lbl] for _, lbl in examples]
    max_len  = max(len(ids) for ids in ids_list)

    padded   = [ids + [PAD_IDX] * (max_len - len(ids)) for ids in ids_list]
    token_ids = torch.tensor(padded,  dtype=torch.long,  device=device)
    label_ids = torch.tensor(labels,  dtype=torch.long,  device=device)
    pad_mask  = (token_ids == PAD_IDX)
    return token_ids, label_ids, pad_mask


def get_data(seed=42):
    raw                              = get_classification_data(n_total=200, seed=seed)
    train_data, val_data, test_data  = split_data(raw, seed=seed)
    word2idx                         = build_vocab(train_data)
    return train_data, val_data, test_data, word2idx
