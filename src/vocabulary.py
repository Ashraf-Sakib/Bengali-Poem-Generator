import pickle
import os
from collections import Counter
from typing import List, Dict, Set, Optional, Tuple, Any


class Vocabulary:
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    START_TOKEN = "<START>"
    END_TOKEN = "<END>"
    NEWLINE_TOKEN = "<LINE>"

    SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, START_TOKEN, END_TOKEN, NEWLINE_TOKEN]

    def __init__(self, min_freq: int = 5):
        self.min_freq = min_freq
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.word_counts: Counter = Counter()
        self.vocab_size: int = 0

    @property
    def pad_idx(self) -> int:
        return self.word2idx.get(self.PAD_TOKEN, 0)

    @property
    def unk_idx(self) -> int:
        return self.word2idx.get(self.UNK_TOKEN, 1)

    @property
    def start_idx(self) -> int:
        return self.word2idx.get(self.START_TOKEN, 2)

    @property
    def end_idx(self) -> int:
        return self.word2idx.get(self.END_TOKEN, 3)

    @property
    def line_idx(self) -> int:
        return self.word2idx.get(self.NEWLINE_TOKEN, 4)

    def build_vocabulary(self, tokenized_corpus: List[List[str]]) -> None:
        self.word_counts = Counter()
        for tokens in tokenized_corpus:
            self.word_counts.update(tokens)

        self.word2idx = {}
        self.idx2word = {}
        for token in self.SPECIAL_TOKENS:
            idx = len(self.word2idx)
            self.word2idx[token] = idx
            self.idx2word[idx] = token

        for word, count in self.word_counts.items():
            if count >= self.min_freq and word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

        self.vocab_size = len(self.word2idx)

    def word_to_idx(self, word: str) -> int:
        return self.word2idx.get(word, self.word2idx[self.UNK_TOKEN])

    def idx_to_word(self, idx: int) -> str:
        return self.idx2word.get(idx, self.UNK_TOKEN)

    def encode(self, tokens: List[str], add_boundaries: bool = False) -> List[int]:
        indices = []
        if add_boundaries:
            indices.append(self.word2idx[self.START_TOKEN])
        for w in tokens:
            indices.append(self.word_to_idx(w))
        if add_boundaries:
            indices.append(self.word2idx[self.END_TOKEN])
        return indices

    def decode(self, indices: List[int], filter_specials: bool = True) -> List[str]:
        tokens = []
        for idx in indices:
            word = self.idx_to_word(idx)
            if filter_specials and word in self.SPECIAL_TOKENS:
                continue
            tokens.append(word)
        return tokens

    def contains(self, word: str) -> bool:
        return word in self.word2idx and word != self.UNK_TOKEN

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump({
                "word2idx": self.word2idx,
                "idx2word": self.idx2word,
                "word_counts": self.word_counts,
                "vocab_size": self.vocab_size,
                "min_freq": self.min_freq
            }, f)

    @classmethod
    def load(cls, filepath: str) -> "Vocabulary":
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Vocabulary file not found: {filepath}")
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        vocab = cls(min_freq=data["min_freq"])
        vocab.word2idx = data["word2idx"]
        vocab.idx2word = data["idx2word"]
        vocab.word_counts = data["word_counts"]
        vocab.vocab_size = data["vocab_size"]
        return vocab

    def get_summary(self) -> Dict[str, Any]:
        pruned_words = sum(1 for cnt in self.word_counts.values() if cnt < self.min_freq)
        return {
            "vocab_size": self.vocab_size,
            "raw_unique_words": len(self.word_counts),
            "special_tokens_count": len(self.SPECIAL_TOKENS),
            "min_frequency": self.min_freq,
            "pruned_rare_words": pruned_words,
            "top_10_words": self.word_counts.most_common(10)
        }