"""
Session 9 — Seq2Seq with Bahdanau Attention
data.py: Tokenize, vocab with SOS/EOS/PAD/UNK, QA dataset, padded batches.

Pipeline:
    get_qa_pairs()                  → [(question, answer), ...]
    split_data(seed=42, 80/10/10)   → train / val / test
    build_vocab(train_only)         → {<PAD>:0, <UNK>:1, <SOS>:2, <EOS>:3, ...}
    QADataset + collate_fn          → (src_padded, tgt_padded, src_mask, tgt_lengths)
"""

import sys
import os
import random
import torch
from torch.utils.data import Dataset, DataLoader

_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_DIR, "..", "..", "..")
sys.path.insert(0, _ROOT)

from code_practice.shared_dataset import get_instruction_pairs

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"

PAD_IDX = 0
UNK_IDX = 1
SOS_IDX = 2
EOS_IDX = 3


def tokenize(text: str) -> list:
    return text.replace(".", " .").replace("?", " ?").split()


def split_data(data, seed=42, train_ratio=0.8, val_ratio=0.1):
    rng  = random.Random(seed)
    data = list(data)
    rng.shuffle(data)
    n       = len(data)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    return data[:n_train], data[n_train:n_train + n_val], data[n_train + n_val:]


def build_vocab(train_data):
    """Shared vocabulary (questions + answers) from train only."""
    word2idx = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX,
                SOS_TOKEN: SOS_IDX, EOS_TOKEN: EOS_IDX}
    for question, answer in train_data:
        for tok in tokenize(question) + tokenize(answer):
            if tok not in word2idx:
                word2idx[tok] = len(word2idx)
    idx2word = {v: k for k, v in word2idx.items()}
    return word2idx, idx2word


def encode(text, word2idx, add_sos=False, add_eos=True):
    tokens = tokenize(text)
    ids = [word2idx.get(t, UNK_IDX) for t in tokens]
    if add_sos:
        ids = [SOS_IDX] + ids
    if add_eos:
        ids = ids + [EOS_IDX]
    return ids


class QADataset(Dataset):
    def __init__(self, data, word2idx):
        self.data     = data
        self.word2idx = word2idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question, answer = self.data[idx]
        src = torch.tensor(encode(question, self.word2idx, add_sos=False, add_eos=True),
                           dtype=torch.long)
        tgt = torch.tensor(encode(answer,   self.word2idx, add_sos=True,  add_eos=True),
                           dtype=torch.long)
        return src, tgt


def collate_fn(batch):
    """
    Returns:
        src_padded [B, T_src]  — question tokens
        tgt_padded [B, T_tgt]  — answer tokens (SOS ... EOS)
        src_mask   [B, T_src]  — 1 for real, 0 for pad
        tgt_lengths [B]        — true tgt lengths
    """
    srcs, tgts = zip(*batch)
    src_lens   = [len(s) for s in srcs]
    tgt_lens   = [len(t) for t in tgts]
    max_src    = max(src_lens)
    max_tgt    = max(tgt_lens)

    src_padded = torch.full((len(batch), max_src), PAD_IDX, dtype=torch.long)
    tgt_padded = torch.full((len(batch), max_tgt), PAD_IDX, dtype=torch.long)
    src_mask   = torch.zeros(len(batch), max_src, dtype=torch.long)

    for i, (s, t) in enumerate(zip(srcs, tgts)):
        src_padded[i, :len(s)] = s
        tgt_padded[i, :len(t)] = t
        src_mask[i, :len(s)]   = 1

    tgt_lengths = torch.tensor(tgt_lens, dtype=torch.long)
    return src_padded, tgt_padded, src_mask, tgt_lengths


def get_loaders(batch_size=16, seed=42):
    raw   = get_instruction_pairs(n_total=100, seed=seed)
    pairs = [(d["instruction"], d["response"]) for d in raw]
    train_data, val_data, test_data = split_data(pairs, seed=seed)

    word2idx, idx2word = build_vocab(train_data)

    train_ds = QADataset(train_data, word2idx)
    val_ds   = QADataset(val_data,   word2idx)
    test_ds  = QADataset(test_data,  word2idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return (
        (train_loader, val_loader, test_loader),
        word2idx, idx2word,
        train_data, val_data, test_data,
    )
