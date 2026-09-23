"""
src/generator.py
-----------------
Poetry generation engine combining Trigram Language Model, Laplace Smoothing,
Word2Vec semantic embeddings, and Bengali Phonetic Rhyme Control to generate
meaningful, rhythmic 4-6 line Bengali poems and chhoras (ছড়া).

Core Traditional NLP Architecture:
1. Word2Vec Thematic Expansion & Semantic Field Filtering
2. Phonetic Couplet Rhyme Planning (AABB scheme: চরণ ১-২ অন্ত্যমিল, চরণ ৩-৪ অন্ত্যমিল)
3. Dominant Trigram Language Model Next-Word Prediction with Laplace Smoothing
4. Candidate Filtering & Logit-Level Repetition Penalty
5. Top 4-6 Candidate Sampling with Softmax Temperature
6. Backtracking & Context Recovery (prevents dead-ends and malformed outputs)
7. Strict Line Length (5-8 words) & Natural Verse Boundary Enforcement
8. Best-of-N (e.g. 20) Candidate Poem Scoring & Reranking System
9. Automatic Modern HTML Showcase Output (output/poem_viewer.html)
"""

import random
import math
import unicodedata
from collections import Counter
from typing import List, Dict, Tuple, Optional, Any, Set

from src.trigram_model import TrigramLanguageModel
from src.smoothing import LaplaceSmoothing
from src.word2vec_model import BengaliWord2Vec
from src.vocabulary import Vocabulary
from src.rhyme_controller import BengaliRhymeController
from src.utils import save_poem_to_html


# Curated poetic domain anchors for common Bengali literary themes
THEME_DOMAIN_ANCHORS: Dict[str, List[str]] = {
    "নদী": ["জল", "সাগর", "তরঙ্গ", "কূল", "তীর", "নৌকা", "মাঝি", "ধারা", "স্রোত", "ভাসে", "অকূল", "বাতাস", "উছল", "বয়ে", "পাড়", "ঘাট", "খেয়া"],
    "বৃষ্টি": ["মেঘ", "মেঘের", "জল", "বাদল", "বর্ষা", "ভিজে", "ভেজা", "আকাশ", "ঝরে", "ঝরেছে", "শীতল", "হাওয়া", "নূপুর", "শ্রাবণ", "বিন্দু"],
    "প্রকৃতি": ["ফুল", "পাখি", "সবুজ", "বন", "অরণ্য", "গাছ", "পাতা", "রোদ", "ছায়া", "সকাল", "ভোর", "শিশির", "বাতাস", "মাঠ", "দিগন্ত", "আকাশ"],
    "ভালোবাসা": ["মন", "হৃদয়", "পরান", "চোখ", "প্রেম", "স্মৃতি", "মায়া", "আশা", "স্বপ্ন", "বেদনা", "নীরব", "মধুর", "কাছে", "চিরকাল", "জীবন"],
    "জীবন": ["পথ", "চলা", "দিন", "রাত্রি", "আলো", "আঁধার", "সুখ", "দুঃখ", "স্বপ্ন", "আশা", "সময়", "কাল", "জগৎ", "সংসার", "মরণ"],
    "মৃত্যু": ["অন্ধকার", "নীরব", "ঘুম", "চিরকাল", "বিদায়", "স্মৃতি", "শূন্য", "মরণ", "প্রাণ", "দেহ", "শেষ", "অবশেষ", "অজানা", "ছায়া"],
    "মা": ["স্নেহ", "আঁচল", "মমতা", "ভালোবাসা", "কোলে", "ডাক", "হাসি", "চোখের", "দুনিয়া", "সংসার", "পরান", "আশীর্বাদ"]
}


class BanglaPoemGenerator:
    """
    High-coherence Bengali Poetry & Chhora Generator using:
    - Primary Trigram Language Model with Laplace Smoothing
    - Phonetic Rhyme Controller with AABB Couplet Planning
    - Thematic Semantic Field Filtering via Word2Vec
    - Top 4-6 Softmax Temperature Sampling
    - Logit-level Repetition Penalty & Bigram Loop Prevention
    - Best-of-N (e.g. 20) Multi-factor Candidate Reranking
    - Automatic Modern HTML Showcase Output
    """

    def __init__(
        self,
        lm: TrigramLanguageModel,
        smoother: LaplaceSmoothing,
        w2v: BengaliWord2Vec,
        vocab: Vocabulary,
        trigram_weight: float = 0.60,
        semantic_weight: float = 0.25,
        rhyme_weight: float = 0.15
    ):
        """
        Initialize the Poem Generator.

        Args:
            lm (TrigramLanguageModel): Trained trigram count model.
            smoother (LaplaceSmoothing): Laplace / Lidstone smoother.
            w2v (BengaliWord2Vec): Word2Vec model for theme expansion and scoring.
            vocab (Vocabulary): Corpus vocabulary with special tokens.
            trigram_weight (float): Weight for trigram fluency (default: 0.60).
            semantic_weight (float): Weight for Word2Vec semantic relevance (default: 0.25).
            rhyme_weight (float): Weight for end-rhyme cadence (default: 0.15).
        """
        self.lm = lm
        self.smoother = smoother
        self.w2v = w2v
        self.vocab = vocab
        self.w_trigram = trigram_weight
        self.w_semantic = semantic_weight
        self.w_rhyme = rhyme_weight

        # Initialize Phonetic Rhyme Controller
        self.rhyme_ctrl = BengaliRhymeController(lm.line_end_counts, w2v=self.w2v)

    def build_thematic_field(self, keyword: str, top_n: int = 35) -> Dict[str, float]:
        """
        Constructs an expanded thematic semantic field around the keyword.
        Combines Word2Vec vector similarity with curated poetic domain anchors.
        """
        keyword_norm = unicodedata.normalize('NFC', keyword.strip())
        field: Dict[str, float] = {keyword_norm: 1.0}

        # 1. Query Word2Vec similarity
        if self.w2v and self.w2v.model:
            sims = self.w2v.get_similar_words(keyword_norm, topn=top_n)
            for w, score in sims:
                w_norm = unicodedata.normalize('NFC', w)
                if len(w_norm) >= 2 and w_norm not in self.vocab.SPECIAL_TOKENS:
                    field[w_norm] = max(field.get(w_norm, 0.0), float(score))

        # 2. Check curated domain anchors for thematic enrichment
        for theme_key, anchors in THEME_DOMAIN_ANCHORS.items():
            if theme_key in keyword_norm or keyword_norm in theme_key:
                for a in anchors:
                    a_norm = unicodedata.normalize('NFC', a)
                    field[a_norm] = max(field.get(a_norm, 0.0), 0.75)

        return field

    def _get_candidate_words(self, w1: str, w2: str) -> Dict[str, int]:
        """
        Gathers valid candidate continuation words.
        1. Prioritizes words observed following (w1, w2) in the trigram model.
        2. If trigram context is sparse, backs off to bigram continuations of w2.
        3. If bigram context is empty, backs off to top frequent poetic unigrams.
        Never includes special tokens (<START>, <END>, <PAD>, <UNK>) or single-consonant artifacts.
        """
        # 1. Trigram continuations
        candidates = dict(self.lm.get_candidate_continuations(w1, w2))

        # 2. Bigram backoff if trigram context is sparse (< 3 continuations)
        if len(candidates) < 3:
            bigram_conts = {
                bg[1]: cnt for bg, cnt in self.lm.bigram_counts.items()
                if bg[0] == w2 and bg[1] not in self.vocab.SPECIAL_TOKENS
            }
            for w, cnt in bigram_conts.items():
                if w not in candidates:
                    candidates[w] = cnt

        # 3. Unigram backoff if still empty
        if not candidates:
            candidates = {
                w: cnt for w, cnt in self.lm.unigram_counts.most_common(50)
                if w not in self.vocab.SPECIAL_TOKENS and w != w2
            }

        # Filter out special tokens, broken single consonants, and self-repeats
        cleaned: Dict[str, int] = {}
        for w, cnt in candidates.items():
            if w in self.vocab.SPECIAL_TOKENS:
                continue
            if len(w) == 1 and w not in {'এ', 'ও', 'ঐ'}:
                continue
            if w.startswith('\u09DF'):  # broken suffix
                continue
            cleaned[w] = cnt

        return cleaned

    def generate_next_token(
        self,
        w1: str,
        w2: str,
        keyword: str,
        thematic_field: Dict[str, float],
        poem_word_counts: Counter,
        line_word_counts: Counter,
        poem_bigrams: Set[Tuple[str, str]],
        current_line_len: int,
        target_line_len: int,
        target_rhyme_word: Optional[str] = None,
        top_k: int = 5,
        temperature: float = 0.7,
        debug: bool = False
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """
        Predicts and samples the next word token:
        1. Retrieves valid continuations.
        2. Calculates smoothed trigram log-probabilities.
        3. Applies thematic semantic field boosting.
        4. Applies logit-level repetition penalty.
        5. Applies rhyme guidance bonus if approaching verse boundary.
        6. Performs Top-k sampling with temperature scaling.
        """
        raw_candidates = self._get_candidate_words(w1, w2)
        if not raw_candidates:
            return None, None

        # Filter out immediate consecutive duplicate words and repeated bigrams
        valid_candidates = []
        for cand, cnt in raw_candidates.items():
            if cand == w2:
                continue
            if (w2, cand) in poem_bigrams:
                continue
            valid_candidates.append(cand)

        if not valid_candidates:
            valid_candidates = [c for c in raw_candidates if c != w2]

        if not valid_candidates:
            return None, None

        # Calculate Laplace smoothed probabilities
        tri_probs = {}
        for cand in valid_candidates:
            p = self.smoother.get_smoothed_trigram_prob(w1, w2, cand)
            tri_probs[cand] = p

        scored_candidates = []
        approaching_end = (current_line_len >= target_line_len - 2)

        for cand in valid_candidates:
            p_tri = tri_probs[cand]
            log_p = math.log(max(p_tri, 1e-10))

            # Thematic relevance
            theme_sim = thematic_field.get(cand, 0.0)
            if theme_sim == 0.0 and self.w2v:
                theme_sim = max(0.0, self.w2v.get_similarity(keyword, cand))

            # Repetition Penalty (Logit subtraction)
            poem_rep = poem_word_counts.get(cand, 0)
            line_rep = line_word_counts.get(cand, 0)
            rep_pen = (poem_rep * 1.8) + (line_rep * 3.5)

            # Rhyme & Ending Guidance Bonus
            rhyme_bonus = 0.0
            if approaching_end:
                if target_rhyme_word:
                    if cand == target_rhyme_word:
                        rhyme_bonus = 4.0  # Exact target rhyme landing
                    elif self.rhyme_ctrl.is_rhyme(cand, target_rhyme_word):
                        rhyme_bonus = 3.2  # Rhyming sibling from same family
                elif cand in self.rhyme_ctrl.valid_endings:
                    rhyme_bonus = 1.2  # Natural poetic line termination

            # Composite logit
            logit = (
                self.w_trigram * log_p +
                self.w_semantic * (theme_sim * 4.0) +
                self.w_rhyme * rhyme_bonus -
                rep_pen
            )

            scored_candidates.append({
                "word": cand,
                "logit": logit,
                "trigram_prob": p_tri,
                "theme_sim": theme_sim,
                "rep_pen": rep_pen,
                "rhyme_bonus": rhyme_bonus
            })

        scored_candidates.sort(key=lambda x: x["logit"], reverse=True)

        # Top-k selection (strictly 4-6 candidates)
        k = max(3, min(top_k, 6, len(scored_candidates)))
        top_cands = scored_candidates[:k]

        words_k = [x["word"] for x in top_cands]
        logits_k = [x["logit"] for x in top_cands]

        # Numerically stabilized softmax temperature sampling
        temp = max(0.1, temperature)
        max_l = max(logits_k)
        scaled = [math.exp((l - max_l) / temp) for l in logits_k]
        total_s = sum(scaled)
        probs_k = [s / total_s for s in scaled]

        chosen_word = random.choices(words_k, weights=probs_k, k=1)[0]
        chosen_info = next(x for x in top_cands if x["word"] == chosen_word)

        debug_info = {
            "context": (w1, w2),
            "top_candidates": top_cands,
            "chosen_word": chosen_word,
            "chosen_prob": chosen_info["trigram_prob"],
            "chosen_theme": chosen_info["theme_sim"],
            "chosen_logit": chosen_info["logit"]
        }

        return chosen_word, debug_info

    def generate_line(
        self,
        starter: str,
        keyword: str,
        thematic_field: Dict[str, float],
        poem_word_counts: Counter,
        poem_bigrams: Set[Tuple[str, str]],
        target_rhyme_word: Optional[str] = None,
        min_words: int = 5,
        max_words: int = 8,
        top_k: int = 5,
        temperature: float = 0.7,
        debug_logs: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """
        Generates a single poetic line strictly constrained between min_words (5) and max_words (8).
        Includes backtracking context recovery and rhyme termination targeting.
        """
        target_len = random.randint(min_words, max_words)

        for attempt in range(25):
            line_tokens = [starter]
            line_word_counts = Counter([starter])
            w1 = '<START>'
            w2 = starter
            backtrack_attempts = 0

            while len(line_tokens) < target_len and backtrack_attempts < 15:
                next_w, debug_step = self.generate_next_token(
                    w1=w1,
                    w2=w2,
                    keyword=keyword,
                    thematic_field=thematic_field,
                    poem_word_counts=poem_word_counts,
                    line_word_counts=line_word_counts,
                    poem_bigrams=poem_bigrams,
                    current_line_len=len(line_tokens),
                    target_line_len=target_len,
                    target_rhyme_word=target_rhyme_word,
                    top_k=top_k,
                    temperature=temperature,
                    debug=(debug_logs is not None)
                )

                if next_w is None:
                    # Backtrack to previous token
                    backtrack_attempts += 1
                    if len(line_tokens) > 1:
                        removed = line_tokens.pop()
                        line_word_counts[removed] -= 1
                        w2 = line_tokens[-1]
                        w1 = line_tokens[-2] if len(line_tokens) >= 2 else '<START>'
                    else:
                        break
                    continue

                if debug_logs is not None and debug_step:
                    debug_logs.append(debug_step)

                line_tokens.append(next_w)
                line_word_counts[next_w] += 1
                w1 = w2
                w2 = next_w

                # Natural line termination with rhyme targeting
                if len(line_tokens) >= min_words:
                    if target_rhyme_word and (w2 == target_rhyme_word or self.rhyme_ctrl.is_rhyme(w2, target_rhyme_word)):
                        break
                    if not target_rhyme_word and w2 in self.rhyme_ctrl.valid_endings and len(line_tokens) >= target_len - 1:
                        break

            if len(line_tokens) >= min_words:
                # Register line tokens in poem memory
                for t in line_tokens:
                    poem_word_counts[t] += 1
                for i in range(len(line_tokens) - 1):
                    poem_bigrams.add((line_tokens[i], line_tokens[i + 1]))
                return line_tokens

        # Fallback padding if strict loop didn't satisfy min_words
        fallbacks = ['ছিল', 'থাকে', 'যায়', 'আমার', 'তুমি', 'সে', 'আকাশে', 'জলে']
        while len(line_tokens) < min_words:
            fb = random.choice([f for f in fallbacks if f not in line_word_counts])
            line_tokens.append(fb)
            poem_word_counts[fb] += 1

        return line_tokens

    def generate_single_poem(
        self,
        keyword: str,
        num_lines: int = 4,
        min_words_per_line: int = 5,
        max_words_per_line: int = 8,
        top_k: int = 5,
        temperature: float = 0.7,
        debug_logs: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Generates one complete 4-6 line candidate poem with intentional
        AABB couplet rhyme planning and thematic line context flow.
        """
        num_lines = max(4, min(num_lines, 6))

        # 1. Expand keyword into thematic semantic field
        thematic_field = self.build_thematic_field(keyword, top_n=35)
        theme_words = list(thematic_field.keys())

        # 2. Plan AABB Rhyme Couplets: Couplet 1 (Line 1 & 2), Couplet 2 (Line 3 & 4)
        couplet_plan = self.rhyme_ctrl.plan_aabb_couplets(
            keyword=keyword,
            theme_words=theme_words,
            w2v=self.w2v,
            num_lines=num_lines
        )

        # 3. Select starting contexts
        theme_starters = [
            s for s in [keyword] + theme_words
            if self.lm.history_2_counts.get(('<START>', s), 0) >= 2
        ]
        poetic_general_starters = [
            'এই', 'আমার', 'তুমি', 'সেখানে', 'যেন', 'কখনো', 'আকাশে', 'নীরব',
            'হৃদয়ে', 'চোখে', 'দূরে', 'আলো', 'বাতাসে', 'একদিন'
        ]

        poem_word_counts = Counter()
        poem_bigrams = set()
        raw_lines = []
        planned_endings = []

        for idx in range(num_lines):
            couplet_idx = idx // 2
            line_in_couplet = idx % 2
            planned_target = None

            if couplet_idx < len(couplet_plan):
                c_pair = couplet_plan[couplet_idx]
                planned_target = c_pair[line_in_couplet]

            planned_endings.append(planned_target)

            # Select line starter
            if idx == 0 and theme_starters:
                starter = theme_starters[0]
            elif idx == 1 and len(theme_starters) > 1:
                starter = theme_starters[1]
            elif idx == 2:
                # Couplet 2 starts with evocative imagery or reflection
                available = [s for s in theme_starters[2:] if s not in poem_word_counts] + poetic_general_starters
                starter = random.choice(available) if available else 'এই'
            else:
                available = [s for s in poetic_general_starters if s not in poem_word_counts]
                starter = random.choice(available) if available else 'আমার'

            line = self.generate_line(
                starter=starter,
                keyword=keyword,
                thematic_field=thematic_field,
                poem_word_counts=poem_word_counts,
                poem_bigrams=poem_bigrams,
                target_rhyme_word=planned_target,
                min_words=min_words_per_line,
                max_words=max_words_per_line,
                top_k=top_k,
                temperature=temperature,
                debug_logs=debug_logs if idx == 0 else None
            )
            raw_lines.append(line)

        # Format lines with Bengali punctuation (comma on odd lines, Dari । on even/last lines)
        formatted_lines = []
        for idx, line in enumerate(raw_lines):
            line_str = " ".join(line)
            if idx == len(raw_lines) - 1 or idx % 2 == 1:
                line_str += "।"
            else:
                line_str += ","
            formatted_lines.append(line_str)

        return {
            "raw_lines": raw_lines,
            "formatted_lines": formatted_lines,
            "poem_text": "\n".join(formatted_lines),
            "word_counts": poem_word_counts,
            "theme_words": theme_words,
            "planned_endings": planned_endings
        }

    def score_poem_candidate(
        self,
        candidate: Dict[str, Any],
        keyword: str,
        thematic_field: Dict[str, float]
    ) -> float:
        """
        Calculates composite score for candidate poem reranking:
        - Trigram Log-Probability (Fluency)
        - Thematic Semantic Coherence
        - Authentic AABB Rhyme Cadence Bonus
        - Word Repetition Penalty
        - Line-Length Balance Penalty
        """
        raw_lines = candidate["raw_lines"]
        total_tokens = sum(len(l) for l in raw_lines)
        if total_tokens == 0:
            return -999.0

        # 1. Trigram Fluency (Average log probability)
        total_log_p = 0.0
        n_transitions = 0
        for line in raw_lines:
            padded = ['<START>', '<START>'] + line + ['<END>']
            for i in range(2, len(padded)):
                w1, w2, w3 = padded[i - 2], padded[i - 1], padded[i]
                p = self.smoother.get_smoothed_trigram_prob(w1, w2, w3)
                total_log_p += math.log(max(p, 1e-12))
                n_transitions += 1

        avg_log_p = (total_log_p / n_transitions) if n_transitions > 0 else -10.0

        # 2. Thematic Semantic Relevance
        flat_words = [w for line in raw_lines for w in line]
        theme_matches = sum(thematic_field.get(w, 0.0) for w in flat_words)
        theme_score = (theme_matches / len(flat_words)) if flat_words else 0.0

        # 3. Rhyme Cadence Evaluation (AABB)
        rhyme_eval = self.rhyme_ctrl.score_poem_rhyme_cadence(raw_lines, scheme='AABB', w2v=self.w2v)
        rhyme_bonus = rhyme_eval["rhyme_score"] * 8.0  # Big bonus for couplet rhymes!

        # 4. Word Repetition Penalty
        word_counts = candidate["word_counts"]
        rep_penalty = 0.0
        for w, cnt in word_counts.items():
            if cnt > 1 and w not in {'এই', 'ও', 'আর', 'যে', 'সে', 'না'}:
                rep_penalty += (cnt - 1) * 2.5

        # 5. Line Length Variance Penalty (Target 6-7 words per line)
        line_lengths = [len(l) for l in raw_lines]
        mean_len = sum(line_lengths) / len(line_lengths)
        len_var = sum((l - mean_len) ** 2 for l in line_lengths) / len(line_lengths)
        len_penalty = math.sqrt(len_var) * 0.4

        composite_score = (
            avg_log_p * 1.0 +
            theme_score * 7.0 +
            rhyme_bonus -
            rep_penalty * 2.0 -
            len_penalty * 1.0
        )

        candidate["scores"] = {
            "total_score": round(composite_score, 3),
            "fluency_log_prob": round(avg_log_p, 3),
            "thematic_relevance": round(theme_score, 4),
            "rhyme_score": rhyme_eval["rhyme_score"],
            "rhyme_bonus": round(rhyme_bonus, 2),
            "repetition_penalty": round(rep_penalty, 2),
            "length_penalty": round(len_penalty, 2),
            "rhyme_details": rhyme_eval["details"]
        }

        return composite_score

    def generate_poem(
        self,
        keyword: str,
        num_lines: int = 4,
        num_candidates: int = 20,
        min_words_per_line: int = 5,
        max_words_per_line: int = 8,
        top_k: int = 5,
        temperature: float = 0.7,
        debug: bool = False
    ) -> Dict[str, Any]:
        """
        Best-of-N Poem Generation Pipeline:
        1. Builds thematic semantic field for keyword.
        2. Generates N (default: 20) candidate poems independently with AABB couplet rhyme planning.
        3. Evaluates candidates using composite fluency, rhyme cadence, and thematic consistency score.
        4. Selects and returns the single highest-scoring poem.
        5. Writes HTML showcase to output/poem_viewer.html.
        """
        num_lines = max(4, min(num_lines, 6))
        debug_logs = [] if debug else None

        thematic_field = self.build_thematic_field(keyword, top_n=35)

        candidates = []
        for i in range(num_candidates):
            cur_debug = debug_logs if (debug and i == 0) else None
            cand = self.generate_single_poem(
                keyword=keyword,
                num_lines=num_lines,
                min_words_per_line=min_words_per_line,
                max_words_per_line=max_words_per_line,
                top_k=top_k,
                temperature=temperature,
                debug_logs=cur_debug
            )
            score = self.score_poem_candidate(cand, keyword, thematic_field)
            candidates.append((score, cand))

        candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, best_poem = candidates[0]

        # Calculate exact NLP metrics
        flat_words = [w for line in best_poem["raw_lines"] for w in line]
        unique_words = len(set(flat_words))
        total_words = len(flat_words)
        ttr = round(unique_words / total_words, 4) if total_words > 0 else 0.0

        # Calculate Perplexities
        line_perplexities = []
        for line in best_poem["raw_lines"]:
            pp = self.smoother.calculate_sentence_perplexity(line)
            line_perplexities.append(round(pp, 2))
        avg_pp = round(sum(line_perplexities) / len(line_perplexities), 2) if line_perplexities else 0.0

        # Rhyme cadence summary
        rhyme_details = best_poem["scores"]["rhyme_details"]
        rhyme_score = best_poem["scores"]["rhyme_score"]

        sim_pairs = self.w2v.get_similar_words(keyword, topn=8) if self.w2v else []

        result_dict = {
            "keyword": keyword,
            "num_lines": num_lines,
            "lines": best_poem["formatted_lines"],
            "poem_text": best_poem["poem_text"],
            "token_lines": best_poem["raw_lines"],
            "word2vec_similar_words": sim_pairs,
            "theme_seeds_used": list(thematic_field.keys())[:10],
            "scoring_details": best_poem["scores"],
            "candidates_evaluated": num_candidates,
            "debug_trace": debug_logs,
            "metrics": {
                "total_words": total_words,
                "unique_words": unique_words,
                "type_token_ratio": ttr,
                "line_perplexities": line_perplexities,
                "average_perplexity": avg_pp,
                "rhyme_cadence_score": rhyme_score,
                "rhyme_scheme": "AABB",
                "average_words_per_line": round(total_words / num_lines, 2)
            }
        }

        # Automatically output HTML showcase
        save_poem_to_html(
            best_poem["formatted_lines"],
            keyword=keyword,
            metadata={
                "total_words": total_words,
                "ttr": ttr,
                "avg_perplexity": avg_pp,
                "rhyme_score": rhyme_score
            },
            rhyme_scheme="AABB"
        )

        return result_dict
