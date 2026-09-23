import pickle
import os
import math
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional, Any, Set
from src.vocabulary import Vocabulary


class TrigramLanguageModel:

    def __init__(self, vocab: Vocabulary):

        self.vocab = vocab
        
        # N-gram count tables
        # unigram_counts[w] = count of word w
        self.unigram_counts: Counter = Counter()
        
        # bigram_counts[(w1, w2)] = count of sequence (w1, w2)
        self.bigram_counts: Dict[Tuple[str, str], int] = Counter()
        
        # bigram_history_counts[w1] = sum of bigrams starting with w1 (i.e. count of w1 as prefix)
        self.history_1_counts: Counter = Counter()
        
        # trigram_counts[(w1, w2)][w3] = count of trigram (w1, w2, w3)
        self.trigram_counts: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
        
        # bigram_history_counts[(w1, w2)] = count of sequence (w1, w2) as prefix
        self.history_2_counts: Counter = Counter()

        # line_end_counts[w] = count of how often word w appears directly before <END>
        self.line_end_counts: Counter = Counter()

        # line_start_counts[w] = count of how often word w begins a verse line
        self.line_start_counts: Counter = Counter()

        self.total_words: int = 0
        self.total_lines: int = 0
        self.is_trained: bool = False

    def train(self, tokenized_corpus: List[List[str]], line_end_counts: Optional[Counter] = None) -> None:

        print("Training scratch Trigram Language Model...")
        
        start_tok = self.vocab.START_TOKEN
        end_tok = self.vocab.END_TOKEN

        if line_end_counts:
            self.line_end_counts = line_end_counts.copy()

        for line in tokenized_corpus:
            if not line:
                continue
            self.total_lines += 1

            # Pad sentence with two start tokens and one end token
            padded_tokens = [start_tok, start_tok] + line + [end_tok]

            # Record line start and ending word
            if line:
                self.line_start_counts[line[0]] += 1
                self.line_end_counts[line[-1]] += 1

            # Collect unigram counts
            for token in padded_tokens:
                self.unigram_counts[token] += 1
                self.total_words += 1

            # Collect bigram counts
            for i in range(len(padded_tokens) - 1):
                bg = (padded_tokens[i], padded_tokens[i + 1])
                self.bigram_counts[bg] += 1
                self.history_1_counts[padded_tokens[i]] += 1

            # Collect trigram counts
            for i in range(len(padded_tokens) - 2):
                w1, w2, w3 = padded_tokens[i], padded_tokens[i + 1], padded_tokens[i + 2]
                self.trigram_counts[(w1, w2)][w3] += 1
                self.history_2_counts[(w1, w2)] += 1

        self.is_trained = True
        unique_trigrams = sum(len(c) for c in self.trigram_counts.values())
        print(f"Trigram LM Training Complete:")
        print(f"  Total Lines Processed : {self.total_lines:,}")
        print(f"  Total Tokens Counted  : {self.total_words:,}")
        print(f"  Unique Unigrams       : {len(self.unigram_counts):,}")
        print(f"  Unique Bigrams        : {len(self.bigram_counts):,}")
        print(f"  Unique Trigrams       : {unique_trigrams:,}")
        print(f"  Unique Line End Words : {len(self.line_end_counts):,}")

    def get_unigram_count(self, w: str) -> int:
        """Returns raw frequency count C(w)."""
        return self.unigram_counts.get(w, 0)

    def get_bigram_count(self, w1: str, w2: str) -> int:
        """Returns raw frequency count C(w1, w2)."""
        return self.bigram_counts.get((w1, w2), 0)

    def get_trigram_count(self, w1: str, w2: str, w3: str) -> int:
        """Returns raw frequency count C(w1, w2, w3)."""
        if (w1, w2) in self.trigram_counts:
            return self.trigram_counts[(w1, w2)].get(w3, 0)
        return 0

    def get_mle_trigram_prob(self, w1: str, w2: str, w3: str) -> float:

        history_count = self.history_2_counts.get((w1, w2), 0)
        if history_count == 0:
            return 0.0
        trigram_count = self.get_trigram_count(w1, w2, w3)
        return trigram_count / history_count

    def get_mle_bigram_prob(self, w1: str, w2: str) -> float:
        history_count = self.history_1_counts.get(w1, 0)
        if history_count == 0:
            return 0.0
        return self.get_bigram_count(w1, w2) / history_count

    def get_candidate_continuations(self, w1: str, w2: str) -> Dict[str, int]:
        return dict(self.trigram_counts.get((w1, w2), {}))

    def save(self, filepath: str) -> None:

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = {
            "unigram_counts": self.unigram_counts,
            "bigram_counts": self.bigram_counts,
            "history_1_counts": self.history_1_counts,
            "trigram_counts": dict(self.trigram_counts),
            "history_2_counts": self.history_2_counts,
            "line_end_counts": self.line_end_counts,
            "line_start_counts": self.line_start_counts,
            "total_words": self.total_words,
            "total_lines": self.total_lines
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        print(f"Trigram LM saved to: {filepath}")

    @classmethod
    def load(cls, filepath: str, vocab: Vocabulary) -> "TrigramLanguageModel":
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Trigram model file not found: {filepath}")
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        lm = cls(vocab=vocab)
        lm.unigram_counts = data["unigram_counts"]
        lm.bigram_counts = data["bigram_counts"]
        lm.history_1_counts = data["history_1_counts"]
        lm.trigram_counts = defaultdict(Counter, data["trigram_counts"])
        lm.history_2_counts = data["history_2_counts"]
        lm.line_end_counts = data.get("line_end_counts", Counter())
        lm.line_start_counts = data.get(
            "line_start_counts",
            Counter(data.get("trigram_counts", {}).get((vocab.START_TOKEN, vocab.START_TOKEN), {}))
        )
        lm.total_words = data["total_words"]
        lm.total_lines = data["total_lines"]
        lm.is_trained = True
        print(f"Trigram LM loaded from {filepath}.")
        return lm