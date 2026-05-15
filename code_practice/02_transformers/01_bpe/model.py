"""
Phase 2, Session 1 — BPE Tokenizer from Scratch
model.py: BPE class — train (count pairs → merge), encode, decode, save, load.

Algorithm:
    TRAIN:
        1. Pre-tokenize corpus into words, count word frequencies
        2. Represent each word as list of chars + </w> end-of-word marker
        3. Repeat until vocab reaches target size:
            a. Count all adjacent pair frequencies (weighted by word freq)
            b. Pick most frequent pair (A, B)
            c. Merge → new token "AB" everywhere
            d. Add "AB" to vocab with its rank
        4. Save: vocab + ordered merge list

    ENCODE (new text):
        1. Pre-tokenize into words
        2. For each word: start with char-list + </w>
            a. Find lowest-rank pair present in current split
            b. Apply that merge
            c. Repeat until no merges apply
        3. Look up token IDs in vocab
"""

import re
import json
import os
from collections import defaultdict


EOW = "</w>"   # end-of-word marker (GPT-2 uses Ġ instead)


class BPE:
    def __init__(self):
        self.vocab   = {}        # token → id
        self.merges  = []        # ordered list of (A, B) merge rules
        self._rank   = {}        # (A, B) → merge rank (for fast encode)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self, corpus: str, vocab_size: int = 300, verbose: bool = True):
        """
        Learn BPE merges from corpus string until vocab reaches vocab_size.
        """
        # Step 1: word frequency count
        word_freqs = self._count_words(corpus)

        # Step 2: initial char-level splits  {"low": ["l","o","w","</w>"], ...}
        splits = {word: list(word[:-len(EOW)]) + [EOW]
                  for word in word_freqs}

        # Initial vocab = all unique chars + EOW
        base_chars = set()
        for chars in splits.values():
            base_chars.update(chars)
        self.vocab = {ch: i for i, ch in enumerate(sorted(base_chars))}

        if verbose:
            print(f"Corpus: {len(corpus):,} chars, {len(word_freqs):,} unique words")
            print(f"Initial vocab (chars): {len(self.vocab)}")
            print(f"Target vocab: {vocab_size}")
            merges_needed = vocab_size - len(self.vocab)
            print(f"Merges needed: {merges_needed}\n")

        # Step 3: merge loop
        while len(self.vocab) < vocab_size:
            pair_freqs = self._count_pairs(splits, word_freqs)
            if not pair_freqs:
                break

            best = max(pair_freqs, key=lambda p: (pair_freqs[p], p))
            merged = best[0] + best[1]

            self.merges.append(best)
            self._rank[best] = len(self.merges) - 1
            self.vocab[merged] = len(self.vocab)

            splits = self._apply_merge(splits, best)

            if verbose and len(self.merges) <= 20:
                print(f"  Merge {len(self.merges):3d}: {best[0]!r:6} + {best[1]!r:8} → {merged!r:12}  "
                      f"(count: {pair_freqs[best]})")

        if verbose:
            print(f"\nFinal vocab size: {len(self.vocab)}")
            print(f"Total merges learned: {len(self.merges)}")

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------
    def encode(self, text: str) -> list:
        """
        Encode text string → list of token IDs.
        Silently skips chars not in vocab (use byte-level BPE to avoid this).
        """
        tokens = []
        for word in self._pretokenize(text):
            word_key = word + EOW
            pieces   = self._encode_word(word_key)
            for p in pieces:
                if p in self.vocab:
                    tokens.append(self.vocab[p])
        return tokens

    def decode(self, ids: list) -> str:
        """Decode token IDs → string. Removes </w> markers."""
        id2tok = {v: k for k, v in self.vocab.items()}
        text   = "".join(id2tok.get(i, "") for i in ids)
        return text.replace(EOW, " ").strip()

    def tokenize(self, text: str) -> list:
        """Return token strings (not IDs) for inspection."""
        tokens = []
        for word in self._pretokenize(text):
            word_key = word + EOW
            tokens.extend(self._encode_word(word_key))
        return tokens

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {"vocab": self.vocab, "merges": self.merges}
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str):
        with open(path) as f:
            data = json.load(f)
        bpe = cls()
        bpe.vocab  = data["vocab"]
        bpe.merges = [tuple(m) for m in data["merges"]]
        bpe._rank  = {tuple(m): i for i, m in enumerate(bpe.merges)}
        return bpe

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _pretokenize(text: str) -> list:
        """Split text into words (simple whitespace + punctuation split)."""
        return re.findall(r"\w+|[^\w\s]", text)

    @staticmethod
    def _count_words(corpus: str) -> dict:
        freqs = defaultdict(int)
        for word in re.findall(r"\w+|[^\w\s]", corpus):
            freqs[word + EOW] += 1
        return dict(freqs)

    @staticmethod
    def _count_pairs(splits: dict, word_freqs: dict) -> dict:
        """Count adjacent pair frequencies, weighted by word frequency."""
        pair_freqs = defaultdict(int)
        for word, freq in word_freqs.items():
            chars = splits[word]
            for i in range(len(chars) - 1):
                pair_freqs[(chars[i], chars[i + 1])] += freq
        return dict(pair_freqs)

    @staticmethod
    def _apply_merge(splits: dict, pair: tuple) -> dict:
        """Apply one merge rule to all word splits."""
        A, B  = pair
        AB    = A + B
        new_splits = {}
        for word, chars in splits.items():
            new_chars = []
            i = 0
            while i < len(chars):
                if i < len(chars) - 1 and chars[i] == A and chars[i + 1] == B:
                    new_chars.append(AB)
                    i += 2
                else:
                    new_chars.append(chars[i])
                    i += 1
            new_splits[word] = new_chars
        return new_splits

    def _encode_word(self, word_key: str) -> list:
        """
        Encode a single word+</w> using learned merges (rank-based greedy).
        Apply lowest-rank merge first, repeat until no merges apply.
        """
        chars = list(word_key[:-len(EOW)]) + [EOW]

        while True:
            # Find the lowest-rank applicable merge
            best_rank = float("inf")
            best_pos  = -1
            for i in range(len(chars) - 1):
                pair = (chars[i], chars[i + 1])
                if pair in self._rank and self._rank[pair] < best_rank:
                    best_rank = self._rank[pair]
                    best_pos  = i

            if best_pos == -1:
                break   # no applicable merges

            A, B  = chars[best_pos], chars[best_pos + 1]
            chars = chars[:best_pos] + [A + B] + chars[best_pos + 2:]

        return chars
