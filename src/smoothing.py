import math
from typing import List, Dict, Tuple, Optional, Any
from src.trigram_model import TrigramLanguageModel
from src.vocabulary import Vocabulary

class LaplaceSmoothing:


    def __init__(self, lm: TrigramLanguageModel, alpha: float = 1.0):
        """
        Initialize Laplace Smoothing.

        Args:
            lm (TrigramLanguageModel): Trained trigram language model.
            alpha (float): Smoothing parameter (alpha=1.0 for Laplace, 0 < alpha < 1.0 for Lidstone).
        """
        self.lm = lm
        self.vocab = lm.vocab
        self.alpha = float(alpha)
        # Total effective vocabulary size
        self.vocab_size = max(self.vocab.vocab_size, len(self.lm.unigram_counts), 1)

    def get_smoothed_trigram_prob(self, w1: str, w2: str, w3: str) -> float:
        """
        Calculates smoothed conditional probability P(w3 | w1, w2):
        
        Case 1: History (w1, w2) has been observed in training data:
            P(w3 | w1, w2) = [C(w1, w2, w3) + alpha] / [C(w1, w2) + alpha * |V|]

        Case 2: History (w1, w2) was never seen (C(w1, w2) == 0):
            Back off to smoothed bigram probability:
            P(w3 | w2) = [C(w2, w3) + alpha] / [C(w2) + alpha * |V|]

        Case 3: If w2 is also unseen (C(w2) == 0):
            Back off to smoothed unigram probability:
            P(w3) = [C(w3) + alpha] / [N + alpha * |V|]
        """
        c_w1_w2_w3 = self.lm.get_trigram_count(w1, w2, w3)
        c_w1_w2 = self.lm.history_2_counts.get((w1, w2), 0)

        if c_w1_w2 > 0:
            numerator = c_w1_w2_w3 + self.alpha
            denominator = c_w1_w2 + (self.alpha * self.vocab_size)
            return numerator / denominator

        c_w2_w3 = self.lm.get_bigram_count(w2, w3)
        c_w2 = self.lm.history_1_counts.get(w2, 0)

        if c_w2 > 0:
            numerator = c_w2_w3 + self.alpha
            denominator = c_w2 + (self.alpha * self.vocab_size)
            return numerator / denominator

        # Case 3: Backoff to smoothed unigram
        c_w3 = self.lm.get_unigram_count(w3)
        numerator = c_w3 + self.alpha
        denominator = self.lm.total_words + (self.alpha * self.vocab_size)
        return numerator / denominator

    def get_next_word_distribution(self, w1: str, w2: str, candidate_words: Optional[List[str]] = None) -> List[Tuple[str, float]]:
        """
        Computes smoothed probability distribution over potential candidate next words.
        If candidate_words is None, considers all observed continuations plus top unigrams.

        Returns:
            List[Tuple[str, float]]: Sorted list of (word, probability) descending.
        """
        if candidate_words is None:
            # Candidates: all words seen after (w1, w2), plus top frequent unigrams
            seen_continuations = set(self.lm.get_candidate_continuations(w1, w2).keys())
            # Add top frequent unigrams for exploratory coverage
            top_unigrams = [w for w, _ in self.lm.unigram_counts.most_common(100)]
            candidates = list(seen_continuations.union(top_unigrams))
        else:
            candidates = candidate_words

        word_probs = []
        for w in candidates:
            prob = self.get_smoothed_trigram_prob(w1, w2, w)
            word_probs.append((w, prob))

        word_probs.sort(key=lambda x: x[1], reverse=True)
        return word_probs

    def compare_mle_vs_smoothed(self, sample_trigrams: List[Tuple[str, str, str]]) -> List[Dict[str, Any]]:
        """
        Produces a comparative table between Unsmoothed MLE and Laplace Smoothed
        probabilities for pedagogical demonstration and evaluation.

        Args:
            sample_trigrams: List of (w1, w2, w3) tuples (some seen, some unseen).

        Returns:
            List of comparison records with counts, MLE prob, and Smoothed prob.
        """
        results = []
        for w1, w2, w3 in sample_trigrams:
            tri_count = self.lm.get_trigram_count(w1, w2, w3)
            bi_count = self.lm.history_2_counts.get((w1, w2), 0)
            p_mle = self.lm.get_mle_trigram_prob(w1, w2, w3)
            p_smooth = self.get_smoothed_trigram_prob(w1, w2, w3)

            results.append({
                "trigram": f"({w1}, {w2}, {w3})",
                "trigram_count": tri_count,
                "history_count": bi_count,
                "p_mle": p_mle,
                "p_smoothed": p_smooth,
                "status": "Seen" if tri_count > 0 else ("Unseen Trigram (Seen History)" if bi_count > 0 else "Unseen History")
            })
        return results

    def calculate_sentence_perplexity(self, sentence_tokens: List[str]) -> float:
        """
        Calculates Perplexity (PP) of a token sequence under the smoothed trigram model.
        
        Perplexity formula:
            PP(W) = exp( - 1/M * sum_{i=1}^M log P(w_i | w_{i-2}, w_{i-1}) )
            Lower perplexity indicates higher linguistic fluency and expectedness.

        Args:
            sentence_tokens: List of word tokens.

        Returns:
            float: Perplexity score.
        """
        start_tok = self.vocab.START_TOKEN
        end_tok = self.vocab.END_TOKEN

        padded = [start_tok, start_tok] + sentence_tokens + [end_tok]
        m = len(padded) - 2
        if m <= 0:
            return float('inf')

        log_prob_sum = 0.0
        for i in range(2, len(padded)):
            w1, w2, w3 = padded[i - 2], padded[i - 1], padded[i]
            p = self.get_smoothed_trigram_prob(w1, w2, w3)
            # Avoid math.log(0) by using a tiny positive epsilon floor
            p = max(p, 1e-15)
            log_prob_sum += math.log(p)

        cross_entropy = - (log_prob_sum / m)
        perplexity = math.exp(cross_entropy)
        return perplexity
