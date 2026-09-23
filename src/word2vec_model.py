"""
src/word2vec_model.py
----------------------
Word2Vec embedding model for Bengali poetry.
Learns dense continuous vector representations that capture semantic and
thematic relationships between words in the SAHITTO corpus.

NLP Concepts Demonstrated:
- Distributed Word Representations (Word Embeddings)
- Continuous Skip-gram vs CBOW architecture
- Semantic Similarity via Cosine Distance in Vector Space
- Thematic Clustering for Creative Text Generation

FIXED:
- Removed the misleading `difflib` morphological fallback that produced
  nonsense matches like 'সময়' → 'সৌম্য'.
- OOV queries now silently return an empty list (single optional notice).
- Input words are NFC-normalized and stripped of zero-width characters.
- score_poem_relevance no longer fuzzy-matches the target keyword.
"""

import os
import re
import unicodedata
from typing import List, Dict, Tuple, Optional, Any
from gensim.models import Word2Vec as GensimWord2Vec


class BengaliWord2Vec:
    """
    Word2Vec embedding manager for Bengali poetry corpus.

    Fixes:
      * Cleaner OOV handling (no misleading fuzzy matches)
      * NFC normalization on every query
      * Optional one-shot debug notice per OOV word (suppressed by default)
    """

    def __init__(
        self,
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 2,
        epochs: int = 20,
        sg: int = 1,  # 1 for Skip-gram, 0 for CBOW
    ):
        """
        Args:
            vector_size: Dimensionality of the dense word vectors (default: 100).
            window:      Context window distance (default: 5).
            min_count:   Ignores words appearing fewer than this many times.
                         Set to 1 on small corpora to maximize coverage.
            epochs:      Training iterations across the corpus.
            sg:          1 = Skip-gram, 0 = CBOW.
        """
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.sg = sg
        self.model: Optional[GensimWord2Vec] = None

        # Silently keep track of OOV queries (for a single diagnostic print)
        self._oov_seen: set = set()
        self._print_oov_notice: bool = False

    # ------------------------------------------------------------
    # Training
    # ------------------------------------------------------------
    def train(self, tokenized_corpus: List[List[str]]) -> None:
        """
        Trains Word2Vec on the tokenized poetry corpus.

        Args:
            tokenized_corpus: List of tokenized verse lines or poems.

        Skip-gram objective:
            L = sum_t sum_{-c <= j <= c, j != 0} log P(w_{t+j} | w_t)
        """
        print(
            f"Training Bengali Word2Vec "
            f"(vector_size={self.vector_size}, window={self.window}, "
            f"min_count={self.min_count}, epochs={self.epochs}, sg={self.sg})..."
        )

        self.model = GensimWord2Vec(
            sentences=tokenized_corpus,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            epochs=self.epochs,
            sg=self.sg,
            workers=4,
            seed=42,
        )
        print(
            f"Word2Vec Training Complete! "
            f"Learned embeddings for {len(self.model.wv):,} unique words."
        )

    # ------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------
    @staticmethod
    def _clean_word(word: str) -> str:
        """
        NFC-normalize and strip zero-width / BOM characters so that
        visually identical query words match the trained vocab.
        """
        if not word or not isinstance(word, str):
            return ""
        word = unicodedata.normalize("NFC", word.strip())
        # NFC splits য় ড় ঢ় into base letter + nukta, but the corpus stores them composed
        word = (word.replace("\u09AF\u09BC", "\u09DF")
                    .replace("\u09A1\u09BC", "\u09DC")
                    .replace("\u09A2\u09BC", "\u09DD"))
        word = re.sub(r"[\u200B-\u200D\uFEFF\u00AD\u2060]", "", word)
        return word

    def _warn_oov_once(self, word: str) -> None:
        """
        Prints an OOV notice at most once per unique word (only if
        self._print_oov_notice is True). Default: silent.
        """
        if not self._print_oov_notice:
            return
        if word in self._oov_seen:
            return
        self._oov_seen.add(word)
        print(f"[Word2Vec] '{word}' is OOV; returning empty result.")

    # ------------------------------------------------------------
    # Similarity lookup
    # ------------------------------------------------------------
    def get_similar_words(self, word: str, topn: int = 10) -> List[Tuple[str, float]]:
        """
        Returns top-N most semantically similar Bengali words.

        fix : No more difflib fallback. If the query word is out of
        vocabulary, an empty list is returned. This prevents nonsense
        matches like 'সময়' → 'সৌম্য'.

        Args:
            word: Query word (e.g. 'নদী', 'বৃষ্টি', 'সময়').
            topn: How many neighbors to return.

        Returns:
            List of (word, cosine_similarity) pairs, descending.
        """
        if self.model is None:
            raise ValueError("Model has not been trained or loaded yet.")

        w = self._clean_word(word)
        if not w:
            return []

        if w in self.model.wv:
            return self.model.wv.most_similar(w, topn=topn)

        self._warn_oov_once(w)
        return []

    def get_similarity(self, word1: str, word2: str) -> float:
        """
        Cosine similarity between two words.
        Returns 0.0 if either word is OOV.
        """
        if self.model is None:
            return 0.0
        w1 = self._clean_word(word1)
        w2 = self._clean_word(word2)
        if w1 in self.model.wv and w2 in self.model.wv:
            return float(self.model.wv.similarity(w1, w2))
        return 0.0

    def contains(self, word: str) -> bool:
        """True if the (cleaned) word exists in the trained Word2Vec vocab."""
        if self.model is None:
            return False
        w = self._clean_word(word)
        return w in self.model.wv

    def vocab_size(self) -> int:
        """Number of unique words the Word2Vec model knows."""
        if self.model is None:
            return 0
        return len(self.model.wv)

    def oov_words(self) -> List[str]:
        return sorted(self._oov_seen)

    def enable_oov_notice(self, flag: bool = True) -> None:
        self._print_oov_notice = flag
    # Poem scoring (used by Best-of-N reranker)
    def score_poem_relevance(self, poem_words: List[str], keyword: str) -> float:
       
        if self.model is None or not poem_words:
            return 0.0

        target = self._clean_word(keyword)
        if target not in self.model.wv:
            return 0.0

        sim_scores: List[float] = []
        for w in poem_words:
            wc = self._clean_word(w)
            if wc in self.model.wv and len(wc) >= 2:
                sim = float(self.model.wv.similarity(target, wc))
                if sim > 0:
                    sim_scores.append(sim)

        if not sim_scores:
            return 0.0

        sim_scores.sort(reverse=True)
        top_half = sim_scores[: max(1, len(sim_scores) // 2)]
        return float(sum(top_half) / len(top_half))


    # Persistence
    def save(self, filepath: str) -> None:
        """Saves trained Word2Vec model to disk."""
        if self.model is None:
            raise ValueError("No model to save.")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        print(f"Word2Vec model successfully saved to: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "BengaliWord2Vec":
        """Loads a pre-trained Word2Vec model from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        wrapper = cls()
        wrapper.model = GensimWord2Vec.load(filepath)
        wrapper.vector_size = wrapper.model.vector_size
        print(
            f"Word2Vec model loaded from {filepath}. "
            f"Vocab: {len(wrapper.model.wv):,} words."
        )
        return wrapper