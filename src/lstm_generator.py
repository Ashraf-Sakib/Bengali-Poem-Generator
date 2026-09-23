
import math
import os
import re
import pickle
import random
import unicodedata
from collections import Counter
from typing import List, Dict, Tuple, Optional, Any, Set

try:
    import numpy as np
except ImportError:            # coherence score is skipped without numpy
    np = None

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.vocabulary import Vocabulary
from src.word2vec_model import BengaliWord2Vec
from src.rhyme_controller import BengaliRhymeController
from src.lstm_model import BanglaLSTMModel
from src.poetic_enhancer import (
    PoeticEnhancer,
    is_bad_token,
    is_grammatical_line,
)

NEG = -1e9


def normalize_bn(text: str) -> str:
    """NFC + re-compose য় ড় ঢ় (plain NFC would decompose them and break vocabulary lookup)."""
    if not text or not isinstance(text, str):
        return ""
    t = unicodedata.normalize('NFC', text)
    return (t.replace('\u09AF\u09BC', '\u09DF')
             .replace('\u09A1\u09BC', '\u09DC')
             .replace('\u09A2\u09BC', '\u09DD'))


class UnknownKeywordError(ValueError):
    """Raised when the keyword (and anything related to it) is unknown to the model."""


POETIC_STARTERS = [
    'এই', 'আমার', 'তুমি', 'সেখানে', 'যেন', 'কখনো', 'আকাশে', 'নীরব',
    'হৃদয়ে', 'চোখে', 'দূরে', 'আলো', 'বাতাসে', 'একদিন'
]

FUNCTION_WORDS = {
    'এই', 'ও', 'আর', 'যে', 'সে', 'না', 'এ', 'ঐ', 'ওই', 'সেই', 'তার', 'আমার',
    'তোমার', 'আমি', 'তুমি', 'কি', 'কী', 'তো', 'যেন', 'তবু', 'তাই', 'কিন্তু',
    'যদি', 'তবে', 'এবং', 'বা', 'আজ', 'আজও', 'হয়', 'হয়ে', 'যায়', 'করে',
    'থেকে', 'দিয়ে', 'ছিল', 'আছে', 'নেই', 'মতো', 'জন্য', 'মধ্যে', 'কোনো', 'কোন',
}

_VOWEL_SIGNS = 'ািীুূৃেৈোৌ'
_SUFFIXES_AFTER_VOWEL = ['র', 'তে', 'কে', 'টি', 'টা', 'ও', 'ই', 'গুলো', 'রা', 'দের']
_SUFFIXES_AFTER_CONSONANT = ['ে', 'ের', 'ি', 'কে', 'টি', 'টা', 'ও', 'ই', 'গুলো', 'রা', 'দের', 'েই', 'েও']
_STRIPPABLE_SUFFIXES = ['গুলো', 'দের', 'ের', 'তে', 'কে', 'টি', 'টা', 'র', 'ে', 'ি']


TOPIC_STATS_PATH = os.path.join("models", "topic_stats.pkl")

VALID_RHYME_SCHEMES = ("AABB", "auto", "none")


class CorpusTopicStats:
    """
    Poem-level co-occurrence statistics learned from the corpus (no hand-written word lists).
    """

    def __init__(self, poems: List[List[List[str]]]):
        self.poem_words: List[frozenset] = [
            frozenset(w for line in poem for w in line) for poem in poems if poem
        ]
        self.n = len(self.poem_words)
        self.df: Counter = Counter()
        for ws in self.poem_words:
            self.df.update(ws)

    def related(
        self, seeds, top_n: int = 30, min_co: int = 3, min_lift: float = 2.0, min_df: int = 5
    ) -> List[Tuple[str, float]]:
        seeds = set(seeds)
        docs = [ws for ws in self.poem_words if seeds & ws]
        if len(docs) < 3 or self.n == 0:
            return []
        co: Counter = Counter()
        for ws in docs:
            co.update(ws)
        out = []
        for w, c in co.items():
            if w in seeds or c < min_co or self.df[w] < min_df:
                continue
            lift = (c / len(docs)) / (self.df[w] / self.n)
            if lift < min_lift:
                continue
            p_joint = c / self.n
            npmi = math.log(lift) / -math.log(p_joint) if p_joint < 1.0 else 0.0
            if npmi > 0:
                out.append((w, npmi))
        out.sort(key=lambda x: x[1], reverse=True)
        return out[:top_n]

    def save(self, path: str = TOPIC_STATS_PATH) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump([tuple(sorted(ws)) for ws in self.poem_words], f)

    @classmethod
    def load(cls, path: str = TOPIC_STATS_PATH) -> Optional["CorpusTopicStats"]:
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            word_sets = pickle.load(f)
        obj = cls([])
        obj.poem_words = [frozenset(ws) for ws in word_sets]
        obj.n = len(obj.poem_words)
        for ws in obj.poem_words:
            obj.df.update(ws)
        return obj


class LSTMPoemGenerator:

    # ---- tunables (all logit-space unless noted) --------------------------------------
    THEME_WEIGHT = 2.0
    THEME_SATURATED_SCALE = 0.25
    MIN_THEME_SIM = 0.40
    MIN_THEME_COUNT = 3

    LINE_REP_PENALTY = 1.0
    FUNC_REP_PENALTY = 1.5
    POEM_REP_PENALTY = 0.8
    BIGRAM_PENALTY = 3.0
    PRONOUN_KEYWORD_FREE_USES = 3        # keyword নিজেই pronoun হলে এতবার পর্যন্ত ফ্রি
    PRONOUN_KEYWORD_PENALTY_SCALE = 0.5  # তারপরের penalty normal content-word penalty-র এই ভগ্নাংশ

    ENDING_BOOST = 1.0
    LINE_TOPIC_BONUS = 1.5
    MIN_END_RATIO = 0.05      # a word may end a line only if it does so >= 2% of its occurrences
    END_RATIO_SMOOTHING = 3.0   # pseudo-count so rare words are not blocked just for lack of evidence

    LINE_ATTEMPTS = 6           # samples per line; the best one is kept

    STARTER_THEME_P_FIRST = 0.7
    STARTER_THEME_P_OTHER = 0.3
    TOPIC_PREFIX_WORDS = 2

    # Rhyme weight reduced so fluency/theme dominate; rhyme still helps tie-break.
    SCORE_WEIGHTS = dict(
        fluency=1.0,
        theme=4.0,
        rhyme=1.5,
        coherence=2.0,  # Word2Vec similarity between consecutive lines (0 .. 1)
        keyword=2.0,
        topic_lines=2.0,
        repetition=0.5,
        length=0.2,
        incomplete=1.0,
    )

    def __init__(
        self,
        model: BanglaLSTMModel,
        vocab: Vocabulary,
        w2v: BengaliWord2Vec,
        line_end_counts: Counter,
        line_start_counts: Optional[Counter] = None,
        topic_stats: Optional[CorpusTopicStats] = None,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for LSTM generation.\n"
                "Install it with: pip install torch"
            )

        self.model = model
        self.vocab = vocab
        self.w2v = w2v
        self.rhyme_ctrl = BengaliRhymeController(line_end_counts, w2v=self.w2v)
        self.device = next(model.parameters()).device
        self.model.eval()

        self.start_id = vocab.word2idx[vocab.START_TOKEN]
        self.end_id = vocab.word2idx.get(vocab.END_TOKEN)
        self.unk_id = vocab.word2idx[vocab.UNK_TOKEN]
        self.line_id = vocab.word2idx.get(getattr(vocab, "NEWLINE_TOKEN", "<LINE>"))
        self.topic_conditioned = bool(getattr(model, "topic_conditioned", False))
        self.end_mode = "model" if self.end_id is not None else "heuristic"

        with torch.no_grad():
            probe, _ = self.model(torch.tensor([[self.start_id]], device=self.device))
        self.V = probe.shape[-1]
        self.static_mask = self._build_static_mask()

        self.valid_end_words: Set[str] = set(getattr(self.rhyme_ctrl, "valid_endings", []) or [])
        self._rhyme_cache: Dict[str, "torch.Tensor"] = {}
        self._stem_forms_cache: Dict[str, Set[str]] = {}
        self._ending_ids: Optional["torch.Tensor"] = None
        self.vocab_counts = getattr(vocab, "word_counts", None) or Counter()
        self.topic_stats = topic_stats if topic_stats is not None else CorpusTopicStats.load()

        # how often each word actually ends a line in the corpus (learned, not hand-written)
        # Smoothed: (line-ends + k*prior) / (occurrences + k), prior = share of all tokens that end a
        # line.  A frequent word that never ends a line (এবং) is blocked; a rare word we have barely
        # seen is not.
        self.end_ratio_counts = getattr(self.rhyme_ctrl, "end_word_counts", None) or Counter()
        ends = self.end_ratio_counts
        self.end_ratio: Dict[str, float] = {}
        if self.vocab_counts and ends:
            prior = sum(ends.values()) / max(sum(self.vocab_counts.values()), 1)
            k = self.END_RATIO_SMOOTHING
            self.end_ratio = {
                w: (ends.get(w, 0) + k * prior) / (c + k) for w, c in self.vocab_counts.items()
            }
            n_ok = sum(
                1 for w, r in self.end_ratio.items()
                if r >= self.MIN_END_RATIO and self.end_ratio_counts.get(w, 0) >= 3
            )
            print(f"[generator] {n_ok} of {len(self.end_ratio)} vocabulary words can end a line "
                  f"(end ratio >= {self.MIN_END_RATIO}, count >= 3)")

        self.line_start_counts = line_start_counts if line_start_counts else Counter()
        starters = [
            (w, c) for w, c in self.line_start_counts.most_common(400)
            if self.vocab.contains(w) and len(w) >= 2 and not is_bad_token(w)
        ]
        if not starters:
            starters = [
                (w, c) for w, c in self.vocab.word_counts.most_common(150)
                if len(w) >= 2 and not is_bad_token(w)
            ]
        self.corpus_starters = [w for w, _ in starters]
        self.starter_weights = [float(c) for _, c in starters]

    # ------------------------------------------------------------------ helpers
    def _w2i(self, word: str) -> Optional[int]:
        idx = self.vocab.word_to_idx(normalize_bn(word))
        return None if idx == self.unk_id else idx

    def _usable_id(self, word: str) -> Optional[int]:
        i = self._w2i(word)
        if i is None or i >= self.V:
            return None
        if float(self.static_mask[i]) < NEG / 2:
            return None
        return i

    def _build_static_mask(self) -> "torch.Tensor":
        vals = [0.0] * self.V
        for tok in self.vocab.SPECIAL_TOKENS:
            i = self.vocab.word2idx.get(tok)
            if i is not None and i < self.V:
                vals[i] = NEG
        n_bad = 0
        for i in range(self.V):
            if vals[i] != 0.0:
                continue
            try:
                w = self.vocab.idx_to_word(i)
            except Exception:
                vals[i] = NEG
                continue
            if is_bad_token(w):
                vals[i] = NEG
                n_bad += 1
        if self.end_id is not None and self.end_id < self.V:
            vals[self.end_id] = 0.0
        self.n_masked_junk = n_bad

        if self.V > 0 and n_bad > self.V * 0.30:
            print(f"⚠ WARNING: {n_bad}/{self.V} tokens masked as junk. "
                  f"Check vocabulary/preprocessing.")

        return torch.tensor(vals, dtype=torch.float, device=self.device)

    def _rhyme_ids(self, target: str) -> "torch.Tensor":
        if target not in self._rhyme_cache:
            ids = []
            for w in self.valid_end_words:
                if self.rhyme_ctrl.is_rhyme(w, target):
                    i = self._w2i(w)
                    if i is not None:
                        ids.append(i)
            self._rhyme_cache[target] = torch.tensor(ids, dtype=torch.long, device=self.device)
        return self._rhyme_cache[target]

    def _get_ending_ids(self) -> "torch.Tensor":
        if self._ending_ids is None:
            ids = [i for i in (self._w2i(w) for w in self.valid_end_words) if i is not None]
            self._ending_ids = torch.tensor(ids, dtype=torch.long, device=self.device)
        return self._ending_ids

    # ------------------------------------------------------------------ line completion
    def _can_end_on(self, word: str) -> bool:
        """True if `word` is a plausible last word of a line, judged from the corpus itself."""
        if not self.end_ratio:                  # no statistics available: do not block anything
            return True
        ratio = self.end_ratio.get(word, 0.0)
        count = self.end_ratio_counts.get(word, 0)
        return ratio >= self.MIN_END_RATIO and count >= 3

    # ------------------------------------------------------------------ keyword + thematic field
    def _forms_of(self, word: str) -> Set[str]:
        word = normalize_bn(word)
        if len(word) < 2:
            return set()
        stems = {word}
        for suf in _STRIPPABLE_SUFFIXES:
            if word.endswith(suf) and len(word) - len(suf) >= 3:
                stem = word[:-len(suf)]
                if self._usable_id(stem) is not None:
                    stems.add(stem)
        forms: Set[str] = set()
        for stem in stems:
            suffixes = _SUFFIXES_AFTER_VOWEL if stem[-1] in _VOWEL_SIGNS else _SUFFIXES_AFTER_CONSONANT
            for suf in [''] + suffixes:
                cand = stem + suf
                if self._usable_id(cand) is not None:
                    forms.add(cand)
        return forms

    def _stem(self, word: str) -> str:
        """Naive suffix-stripped stem, used only for repetition grouping
        (NOT a vocab lookup — আকাশ/আকাশের/আকাশে সব একই stem এ পড়বে)."""
        w = normalize_bn(word)
        for suf in sorted(_STRIPPABLE_SUFFIXES, key=len, reverse=True):
            if w.endswith(suf) and len(w) - len(suf) >= 3:
                return w[:-len(suf)]
        return w

    def _forms_for_stem(self, stem: str) -> Set[str]:
        if stem not in self._stem_forms_cache:
            self._stem_forms_cache[stem] = self._forms_of(stem) | {stem}
        return self._stem_forms_cache[stem]

    def _resolve_keyword(self, raw_keyword: str) -> Tuple[str, Set[str], List[str]]:
        warnings: List[str] = []
        kw = normalize_bn(raw_keyword.strip())
        if not kw:
            raise UnknownKeywordError("কোনো শব্দ দেওয়া হয়নি।")

        words = kw.split()
        if any(re.search(r'[^\u0980-\u09FF]', w) for w in words):
            raise UnknownKeywordError(
                f"'{raw_keyword}' বাংলা শব্দ নয়। মডেল শুধু বাংলা শব্দ বোঝে, "
                f"তাই বাংলায় একটি শব্দ লিখুন (যেমন: পাহাড়, নদী)।")

        forms: Set[str] = set()
        for w in words:
            forms |= self._forms_of(w)
        return kw, forms, warnings

    def _build_field(
        self, keyword: str, forms: Set[str], top_n: int = 35
    ) -> Tuple[Dict[str, float], float]:
        kw = normalize_bn(keyword.strip())
        field: Dict[str, float] = {}
        seen_total = 0

        def add(word: str, weight: float):
            nonlocal seen_total
            w = normalize_bn(word)
            if len(w) < 2:
                return
            seen_total += 1
            if self._usable_id(w) is None:
                return
            field[w] = max(field.get(w, 0.0), float(weight))

        for f in sorted(forms):
            add(f, 0.65)

        seeds = {kw} | {normalize_bn(w) for w in kw.split()} | set(forms)
        for seed in seeds:
            if not (self.w2v and getattr(self.w2v, "model", None)):
                break
            try:
                pairs = self.w2v.get_similar_words(seed, topn=top_n) or []
            except Exception:
                pairs = []
            for w, score in pairs:
                w = normalize_bn(w)
                if score < self.MIN_THEME_SIM:
                    continue
                if self.vocab_counts and self.vocab_counts.get(w, 0) < self.MIN_THEME_COUNT:
                    continue
                add(w, score)

        if self.topic_stats is not None:
            for w, npmi in self.topic_stats.related(seeds, top_n=top_n):
                add(w, 0.4 + 0.5 * min(1.0, npmi / 0.6))

        coverage = len(field) / seen_total if seen_total else 0.0
        return field, coverage

    def build_thematic_field(self, keyword: str, top_n: int = 35) -> Dict[str, float]:
        kw, forms, _ = self._resolve_keyword(keyword)
        return self._build_field(kw, forms, top_n)[0]

    def _theme_bias(self, field: Dict[str, float]) -> "torch.Tensor":
        bias = torch.zeros(self.V, device=self.device)
        for w, s in field.items():
            i = self._w2i(w)
            if i is not None and i < self.V:
                bias[i] = max(float(bias[i]), float(s))
        return bias

    def _on_topic(self, word: str, field: Dict[str, float]) -> bool:
        return field.get(word, 0.0) >= self.MIN_THEME_SIM

    # ------------------------------------------------------------------ logit shaping
    def _adjust_logits(
        self,
        raw: "torch.Tensor",
        tokens: List[str],
        poem_counts: Counter,
        poem_bigrams: Set[Tuple[str, str]],
        theme_bias: "torch.Tensor",
        field: Dict[str, float],
        min_words: int,
        target_len: int,
        target_rhyme_word: Optional[str],
        avoid_word: Optional[str],
    ) -> "torch.Tensor":
        hits = sum(1 for t in tokens if self._on_topic(t, field))
        scale = self.THEME_WEIGHT if hits < 2 else self.THEME_WEIGHT * self.THEME_SATURATED_SCALE
        l = raw + self.static_mask + scale * theme_bias
        n = len(tokens)

        # <END> is left to the LSTM (it learned where lines stop); we only forbid it when the line
        # is too short or the last word never ends a line in the corpus.
        if self.end_id is not None:
            can_end = (n >= min_words) and self._can_end_on(tokens[-1]) and (self.end_mode != "heuristic")
            if not can_end:
                l[self.end_id] = NEG

        # never repeat the previous word
        last_i = self._w2i(tokens[-1])
        if last_i is not None:
            l[last_i] = NEG

        # repetition inside the line
        for w, c in Counter(tokens).items():
            i = self._w2i(w)
            if i is not None:
                pen = self.FUNC_REP_PENALTY if w in FUNCTION_WORDS else self.LINE_REP_PENALTY
                l[i] -= pen * c

        # repetition across the poem (content words only) — stem-aware, so
        # আকাশ/আকাশের/আকাশে ইত্যাদি একই lemma-র ভিন্ন রূপ একসাথে penalize হয়
        pronoun_stems = getattr(self, "_active_pronoun_stems", None) or set()
        for stem, c in poem_counts.items():
            if stem in FUNCTION_WORDS:
                if stem in pronoun_stems and c > self.PRONOUN_KEYWORD_FREE_USES:
                    # keyword নিজেই pronoun হলে: ফ্রি quota শেষ হলে হালকা penalty,
                    # unlimited exemption না দিয়ে
                    over = c - self.PRONOUN_KEYWORD_FREE_USES
                    for form in self._forms_for_stem(stem):
                        i = self._w2i(form)
                        if i is not None:
                            l[i] -= self.POEM_REP_PENALTY * self.PRONOUN_KEYWORD_PENALTY_SCALE * over
                continue
            for form in self._forms_for_stem(stem):
                i = self._w2i(form)
                if i is not None:
                    l[i] -= self.POEM_REP_PENALTY * c

        # bigram loop prevention (cross-line bigrams included)
        last = tokens[-1]
        for a, b in poem_bigrams:
            if a == last:
                i = self._w2i(b)
                if i is not None:
                    l[i] -= self.BIGRAM_PENALTY

        # ---- Rhyme steering touches ONLY the final line slot ----
        # (When target_rhyme_word is None we only add a mild "this is a good line-ending" boost,
        #  which happens in every rhyme_scheme.)
        if self._is_final_slot(n, target_len, target_rhyme_word):
            if avoid_word:
                a = self._w2i(avoid_word)
                if a is not None:
                    l[a] = NEG
            constrained = False
            ids = self._rhyme_ids(target_rhyme_word)
            t = self._w2i(target_rhyme_word)
            if t is not None and target_rhyme_word != avoid_word:
                ids = torch.cat([ids, torch.tensor([t], dtype=torch.long, device=self.device)])
            if ids.numel():
                keep = torch.full_like(l, NEG)
                keep[ids] = 0.0
                constrained_l = l + keep
                if float(constrained_l.max()) > NEG / 2:
                    l = constrained_l
                    constrained = True
            if not constrained:
                ends = self._get_ending_ids()
                if ends.numel():
                    l[ends] += self.ENDING_BOOST
        elif not target_rhyme_word and n >= target_len - 1:
            ends = self._get_ending_ids()
            if ends.numel():
                l[ends] += self.ENDING_BOOST

        return l

    @staticmethod
    def _is_final_slot(n_tokens: int, target_len: int, target_rhyme_word: Optional[str]) -> bool:
        return bool(target_rhyme_word) and n_tokens == target_len - 1

    @staticmethod
    def _sample_from_logits(
        l: "torch.Tensor",
        temperature: float,
        top_k: int,
        top_p: float,
        forced_ids: Optional[List[int]] = None,
    ) -> int:
        l = l / max(temperature, 0.1)
        if top_k and top_k > 0:
            k = min(top_k, l.size(-1))
            kth = torch.topk(l, k).values[-1]
            mask = l < kth
            if forced_ids:
                for fid in forced_ids:
                    if 0 <= fid < l.size(-1):
                        mask[fid] = False
            l = torch.where(mask, torch.full_like(l, NEG), l)
        probs = F.softmax(l, dim=-1)
        if top_p < 1.0:
            sp, si = torch.sort(probs, descending=True)
            cum = torch.cumsum(sp, dim=0)
            sp = sp.masked_fill((cum - sp) > top_p, 0.0)
            probs = torch.zeros_like(probs).scatter_(0, si, sp)
            total = probs.sum()
            if total > 0:
                probs = probs / total
        if (not torch.isfinite(probs).all()) or probs.sum() <= 0:
            return int(torch.argmax(l))
        return int(torch.multinomial(probs, 1))

    # ------------------------------------------------------------------ line generation
    @torch.no_grad()
    def _prime(self, starter: str, context=None):
        starter_id = self.vocab.word_to_idx(starter)
        x = torch.tensor([[self.start_id, starter_id]], dtype=torch.long, device=self.device)
        logits, hidden = self.model(x, context)
        return logits[0, -1], hidden

    @torch.no_grad()
    def _topic_context(self, topic: Dict[str, Any]):
        words: List[str] = []
        kw = topic["keyword"].split()[0] if topic["keyword"] else ""
        related = [w for w in sorted(topic["field"], key=topic["field"].get, reverse=True)
                   if w not in topic["forms"] and w != kw]
        ordered = ([kw] if kw else []) + related
        for w in ordered:
            if w not in words and self._usable_id(w) is not None:
                words.append(w)
            if len(words) >= self.TOPIC_PREFIX_WORDS:
                break
        if not words:
            return None
        ids = [self.vocab.word_to_idx(w) for w in words]
        if self.line_id is not None:
            ids.append(self.line_id)
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        _, hidden = self.model(x, None)
        return hidden

    @torch.no_grad()
    def _advance_context(self, context, tokens: List[str]):
        ids = [self.start_id] + [self.vocab.word_to_idx(t) for t in tokens]
        if self.end_id is not None:
            ids.append(self.end_id)
        if self.line_id is not None:
            ids.append(self.line_id)
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        _, hidden = self.model(x, context)
        return hidden

    def _step(self, tid: int, hidden):
        x = torch.tensor([[tid]], dtype=torch.long, device=self.device)
        logits, hidden = self.model(x, hidden)
        return logits[0, 0], hidden

    @torch.no_grad()
    def _sample_line(
        self, starter, theme_bias, field, poem_counts, poem_bigrams, target_rhyme_word, avoid_word,
        min_words, max_words, top_k, top_p, temperature, context=None,
    ) -> Tuple[List[str], bool]:
        raw, hidden = self._prime(starter, context)
        tokens = [starter]
        ended = False
        target_len = random.randint(min_words, max_words)

        while len(tokens) < max_words:
            n_before = len(tokens)
            l = self._adjust_logits(raw, tokens, poem_counts, poem_bigrams, theme_bias, field,
                                    min_words, target_len, target_rhyme_word, avoid_word)

            tid = self._sample_from_logits(l, temperature, top_k, top_p)

            if self.end_id is not None and tid == self.end_id:
                ended = True
                break

            word = self.vocab.idx_to_word(tid)
            tokens.append(word)

            if self._is_final_slot(n_before, target_len, target_rhyme_word):
                ended = True                 # the line stops right after its rhyme word
                break

            if (self.end_mode == "heuristic" and len(tokens) >= target_len
                    and word in self.valid_end_words):
                ended = True
                break

            raw, hidden = self._step(tid, hidden)

        return tokens, ended

    @torch.no_grad()
    def _beam_search_line(
        self, starter, theme_bias, field, poem_counts, poem_bigrams, target_rhyme_word, avoid_word,
        min_words, max_words, temperature, beam_width: int = 5, context=None,
    ) -> Tuple[List[str], bool]:
        raw, hidden = self._prime(starter, context)
        target_len = random.randint(min_words, max_words)
        beams = [dict(tokens=[starter], score=0.0, hidden=hidden, raw=raw, ended=False)]

        def norm(score, tokens, ended):
            return score / (len(tokens) + (1 if ended else 0))

        for _ in range(max_words - 1):
            if all(b["ended"] for b in beams):
                break
            cands = []
            for b in beams:
                if b["ended"]:
                    cands.append(dict(parent=b, tid=None, score=b["score"],
                                      tokens=b["tokens"], ended=True))
                    continue
                n_before = len(b["tokens"])
                adj = self._adjust_logits(b["raw"], b["tokens"], poem_counts, poem_bigrams,
                                          theme_bias, field, min_words, target_len,
                                          target_rhyme_word, avoid_word)
                real_lp = F.log_softmax(b["raw"], dim=-1)
                final_slot = self._is_final_slot(n_before, target_len, target_rhyme_word)
                for tid in torch.topk(adj, min(beam_width * 2, adj.numel())).indices.tolist():
                    if adj[tid] <= NEG / 2:
                        continue
                    s = b["score"] + real_lp[tid].item()
                    if self.end_id is not None and tid == self.end_id:
                        cands.append(dict(parent=b, tid=None, score=s,
                                          tokens=b["tokens"], ended=True))
                    else:
                        w = self.vocab.idx_to_word(tid)
                        cands.append(dict(parent=b, tid=tid, score=s,
                                          tokens=b["tokens"] + [w], ended=final_slot))
            if not cands:
                break
            cands.sort(key=lambda c: norm(c["score"], c["tokens"], c["ended"]), reverse=True)

            new_beams = []
            for c in cands[:beam_width]:
                if c["ended"]:
                    p = c["parent"]
                    new_beams.append(dict(tokens=c["tokens"], score=c["score"],
                                          hidden=p["hidden"], raw=p["raw"], ended=True))
                else:
                    r, h = self._step(c["tid"], c["parent"]["hidden"])
                    new_beams.append(dict(tokens=c["tokens"], score=c["score"],
                                          hidden=h, raw=r, ended=False))
            beams = new_beams

        norms = torch.tensor([norm(b["score"], b["tokens"], b["ended"]) for b in beams])
        probs = F.softmax(norms / max(0.3 * temperature, 0.05), dim=-1)
        pick = int(torch.multinomial(probs, 1))
        return beams[pick]["tokens"], beams[pick]["ended"]

    @staticmethod
    def _postprocess_line(tokens: List[str]) -> List[str]:
        out: List[str] = []
        for w in tokens:
            if is_bad_token(w):
                continue
            if out and out[-1] == w:
                continue
            out.append(w)
        return out

    # ------------------------------------------------------------------ scoring
    @torch.no_grad()
    def _line_logprob(self, tokens: List[str]) -> float:
        use_end = self.end_id is not None and self.end_mode == "model"
        ids = [self.start_id] + list(self.vocab.encode(tokens)) + ([self.end_id] if use_end else [])
        if len(ids) < 2:
            return -15.0
        x = torch.tensor([ids[:-1]], dtype=torch.long, device=self.device)
        logits, _ = self.model(x)
        lp = F.log_softmax(logits[0], dim=-1)
        tgt = torch.tensor(ids[1:], dtype=torch.long, device=self.device)
        return lp.gather(1, tgt.unsqueeze(1)).mean().item()

    @torch.no_grad()
    def _poem_logprob(self, raw_lines: List[List[str]]) -> float:
        """
        Mean log-prob per token for the whole poem, with cross-line context.
        The trivially predictable <START> and <LINE> targets are left out, so the number is
        comparable with per-line values (<END> stays: deciding where a line stops is a real
        prediction).
        """
        if not raw_lines:
            return -15.0
        ids: List[int] = []
        for line in raw_lines:
            if not line:
                continue
            ids.append(self.start_id)
            ids.extend(self.vocab.encode(line))
            if self.end_id is not None:
                ids.append(self.end_id)
            if self.line_id is not None:
                ids.append(self.line_id)
        if len(ids) < 2:
            return -15.0
        x = torch.tensor([ids[:-1]], dtype=torch.long, device=self.device)
        logits, _ = self.model(x)
        lp = F.log_softmax(logits[0], dim=-1)
        tgt_ids = ids[1:]
        tgt = torch.tensor(tgt_ids, dtype=torch.long, device=self.device)
        vals = lp.gather(1, tgt.unsqueeze(1)).squeeze(1).tolist()
        skip = {self.start_id, self.line_id}
        kept = [v for v, t in zip(vals, tgt_ids) if t not in skip]
        return sum(kept) / len(kept) if kept else -15.0

    def _score_line(self, tokens, ended, field, partner_last, context_score: Optional[float] = None) -> float:
        avg_lp = context_score if context_score is not None else self._line_logprob(tokens)
        n = max(len(tokens), 1)
        theme_frac = sum(1 for t in tokens if self._on_topic(t, field)) / n
        has_topic = 1.0 if theme_frac > 0 else 0.0
        rhyme = 0.0
        if partner_last:
            rhyme = 1.0 if self.rhyme_ctrl.is_rhyme(tokens[-1], partner_last) else 0.0
        gram = 1.0 if is_grammatical_line(tokens, w2v=self.w2v, function_words=FUNCTION_WORDS) else 0.0
        return (avg_lp
                + 2.0 * rhyme
                + 0.5 * gram
                + 2.0 * theme_frac
                + self.LINE_TOPIC_BONUS * has_topic
                - (0.0 if ended else 0.5))

    def _is_valid_line(self, tokens: List[str]) -> bool:
        if len(tokens) < 4:
            return False
        if not self._can_end_on(tokens[-1]):
            return False
        if is_bad_token(tokens[-1]):
            return False
        return True

    def _generate_best_line(
        self, starter, field, theme_bias, poem_counts, poem_bigrams,
        target_rhyme_word, avoid_word, partner_last,
        min_words, max_words, top_k, top_p, temperature, use_beam_search, attempts,
        context=None,
    ) -> Tuple[List[str], bool]:
        best = None
        valid = None

        def _try_sampling(temp):
            nonlocal best, valid
            if use_beam_search:
                toks, ended = self._beam_search_line(
                    starter, theme_bias, field, poem_counts, poem_bigrams, target_rhyme_word,
                    avoid_word, min_words, max_words, temp, context=context)
            else:
                toks, ended = self._sample_line(
                    starter, theme_bias, field, poem_counts, poem_bigrams, target_rhyme_word,
                    avoid_word, min_words, max_words, top_k, top_p, temp, context=context)
            toks = self._postprocess_line(toks)
            if not toks:
                return
            score = self._score_line(toks, ended, field, partner_last)
            if self._is_valid_line(toks):
                if valid is None or score > valid[0]:
                    valid = (score, toks, ended)
            if best is None or score > best[0]:
                best = (score, toks, ended)

        for _ in range(max(1, attempts)):
            _try_sampling(temperature)          # keep sampling: the best-scoring valid line wins

        chosen = valid if valid is not None else best
        if chosen is None:
            return [starter], False
        return chosen[1], chosen[2]

    def _pick_starter(self, theme_words: List[str], used: Set[str], line_idx: int) -> str:
        theme_pool = []
        for w in theme_words:
            if w not in used and w not in theme_pool and self._usable_id(w) is not None:
                theme_pool.append(w)

        p_theme = self.STARTER_THEME_P_FIRST if line_idx == 0 else self.STARTER_THEME_P_OTHER
        if theme_pool and random.random() < p_theme:
            return random.choice(theme_pool[:5 if line_idx == 0 else 12])

        corpus = [(w, c) for w, c in zip(self.corpus_starters, self.starter_weights) if w not in used]
        if corpus and random.random() > 0.2:
            words, weights = zip(*corpus)
            return random.choices(words, weights=weights, k=1)[0]
        if theme_pool:
            return random.choice(theme_pool)
        fallback = [w for w in POETIC_STARTERS if self._w2i(w) is not None and w not in used]
        return random.choice(fallback) if fallback else self.corpus_starters[0]

    # ------------------------------------------------------------------ poem generation
    def _format_lines(self, raw_lines: List[List[str]]) -> List[str]:
        out = []
        for idx, line in enumerate(raw_lines):
            s = " ".join(line)
            is_last = idx == len(raw_lines) - 1
            ends_well = bool(line) and self._can_end_on(line[-1])
            if is_last or (ends_well and idx % 2 == 1):
                s += "।"
            else:
                s += ","
            out.append(s)
        return out

    def _prepare_topic(self, keyword: str) -> Dict[str, Any]:
        resolved, forms, warnings = self._resolve_keyword(keyword)
        field, coverage = self._build_field(resolved, forms)
        if not field:
            raise UnknownKeywordError(
                f"'{keyword}' শব্দটি মডেলের অভিধানে নেই এবং কাছাকাছি কোনো শব্দও পাওয়া যায়নি। "
                f"অন্য একটি শব্দ দিন (যেমন: পাহাড়, নদী, আকাশ)।")
        if self.topic_stats is None:
            warnings.append(
                "models/topic_stats.pkl পাওয়া যায়নি; শুধু Word2Vec দিয়ে বিষয় ধরা হচ্ছে। "
                "`python train_lstm_only.py --counts-only` চালান।")
        if not forms:
            warnings.append(
                f"'{resolved}' শব্দটি মডেল সরাসরি চেনে না; কাছাকাছি শব্দ দিয়ে কবিতা লেখা হয়েছে।")

        # keyword নিজে (বা তার কোনো form) pronoun/function word হলে, সেই stem-গুলোকে
        # unlimited repetition-exemption থেকে সরিয়ে soft-cap দাও (নাহলে "তুমি"/"আমি"
        # keyword দিলে পুরো কবিতা pronoun দিয়ে flood হয়ে যায়)
        pronoun_stems: Set[str] = set()
        if resolved in FUNCTION_WORDS or (forms & FUNCTION_WORDS):
            pronoun_stems = {self._stem(resolved)} | {self._stem(f) for f in forms}
            warnings.append(
                f"'{resolved}' একটি সর্বনাম-জাতীয় শব্দ; বেশি ব্যবহৃত হলে স্বয়ংক্রিয়ভাবে "
                f"সংযত করা হবে যাতে কবিতাটা pronoun-এ ভরে না যায়।")

        return dict(keyword=resolved, forms=forms, field=field, coverage=coverage,
                    warnings=warnings, pronoun_stems=pronoun_stems)

    # ------------------------------------------------------------------ AABB planning
    def _plan_rhyme(
        self,
        rhyme_scheme: str,
        keyword: str,
        theme_words: List[str],
        num_lines: int,
    ) -> List:
        """Returns a list of couples [(w1, w2), ...]. Empty when scheme is 'none'."""
        if rhyme_scheme == "none":
            return []
        try:
            return self.rhyme_ctrl.plan_aabb_couplets(
                keyword=keyword, theme_words=theme_words, w2v=self.w2v, num_lines=num_lines)
        except Exception:
            return []

    def _rhyme_target_for_line(
        self,
        rhyme_scheme: str,
        idx: int,
        plan: List,
        raw_lines: List[List[str]],
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Decide the rhyme target for line `idx`.
        Returns (target_rhyme_word, avoid_word, partner_last).
        """
        if rhyme_scheme == "none":
            return None, None, None

        c_idx, pos = divmod(idx, 2)
        planned = None
        if c_idx < len(plan):
            try:
                planned = plan[c_idx][pos]
            except Exception:
                planned = None

        target = planned
        avoid_word = None
        partner_last = None

        if pos == 1 and raw_lines:
            prev_line = raw_lines[-1]
            if not prev_line:
                return target, None, None
            prev_last = prev_line[-1]
            prev_is_complete = self._can_end_on(prev_last)

            if rhyme_scheme == "AABB":
                # Force rhyme even if the previous line was incomplete (old behaviour)
                partner_last, avoid_word = prev_last, prev_last
                if not (planned and self.rhyme_ctrl.is_rhyme(prev_last, planned)):
                    target = prev_last
            elif rhyme_scheme == "auto":
                # Rhyme only when it feels safe (previous line looks complete)
                if prev_is_complete:
                    partner_last, avoid_word = prev_last, prev_last
                    if not (planned and self.rhyme_ctrl.is_rhyme(prev_last, planned)):
                        target = prev_last
                else:
                    target = None
        return target, avoid_word, partner_last

    #  generation
    def generate_single_poem(
        self,
        keyword: str,
        num_lines: int = 4,
        min_words_per_line: int = 5,
        max_words_per_line: int = 8,
        top_k: int = 20,
        top_p: float = 0.90,
        temperature: float = 0.85,
        use_beam_search: bool = False,
        line_attempts: Optional[int] = None,
        topic: Optional[Dict[str, Any]] = None,
        carry_context: bool = True,
        rhyme_scheme: str = "auto",
    ) -> Dict[str, Any]:
        if rhyme_scheme not in VALID_RHYME_SCHEMES:
            rhyme_scheme = "auto"

        num_lines = max(2, min(num_lines, 6))
        topic = topic or self._prepare_topic(keyword)
        keyword = topic["keyword"]
        field = topic["field"]
        self._active_pronoun_stems: Set[str] = topic.get("pronoun_stems") or set()
        theme_bias = self._theme_bias(field)
        theme_words = sorted(field, key=field.get, reverse=True)
        attempts = line_attempts or self.LINE_ATTEMPTS

        plan = self._plan_rhyme(rhyme_scheme, keyword, theme_words, num_lines)

        poem_counts: Counter = Counter()
        poem_bigrams: Set[Tuple[str, str]] = set()
        raw_lines: List[List[str]] = []
        line_info: List[Dict[str, Any]] = []
        used_starters: Set[str] = set()
        base_context = self._topic_context(topic) if self.topic_conditioned else None
        context = base_context

        for idx in range(num_lines):
            target, avoid_word, partner_last = self._rhyme_target_for_line(
                rhyme_scheme, idx, plan, raw_lines)

            starter = self._pick_starter(theme_words, used_starters, idx)
            toks, ended = self._generate_best_line(
                starter, field, theme_bias, poem_counts, poem_bigrams, target, avoid_word,
                partner_last, min_words_per_line, max_words_per_line, top_k, top_p,
                temperature, use_beam_search, attempts, context=context)

            if carry_context:
                context = self._advance_context(context, toks)

            used_starters.add(starter)
            for t in toks:
                poem_counts[self._stem(t)] += 1
            if raw_lines and toks:
                poem_bigrams.add((raw_lines[-1][-1], toks[0]))
            for a, b in zip(toks, toks[1:]):
                poem_bigrams.add((a, b))
            raw_lines.append(toks)
            line_info.append({
                "starter": starter,
                "ended": ended,
                "target_rhyme": target,
                "partner_last": partner_last,
            })

        formatted = self._format_lines(raw_lines)
        return {
            "raw_lines": raw_lines,
            "formatted_lines": formatted,
            "poem_text": "\n".join(formatted),
            "word_counts": poem_counts,
            "theme_words": theme_words,
            "thematic_field": field,
            "thematic_coverage": round(topic["coverage"], 3),
            "keyword_forms": topic["forms"],
            "line_info": line_info,
            "rhyme_scheme": rhyme_scheme,
            "pronoun_stems": self._active_pronoun_stems,
        }

    def _line_vec(self, line: List[str]):
        """Mean Word2Vec vector of a line's content words (None if unavailable)."""
        if np is None or not (self.w2v and getattr(self.w2v, "model", None)):
            return None
        try:
            wv = self.w2v.model.wv
            vs = [wv[w] for w in line if w not in FUNCTION_WORDS and w in wv]
        except Exception:
            return None
        return np.mean(vs, axis=0) if vs else None

    def _coherence(self, raw_lines: List[List[str]]) -> float:
        """Average cosine similarity between consecutive lines (0 when it cannot be computed)."""
        vecs = [v for v in (self._line_vec(l) for l in raw_lines) if v is not None]
        if len(vecs) < 2:
            return 0.0
        def cos(a, b):
            return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
        sims = [max(0.0, cos(a, b)) for a, b in zip(vecs, vecs[1:])]
        return sum(sims) / len(sims)

    def _avg_perplexity(self, raw_lines: List[List[str]]) -> float:
        return math.exp(min(-self._poem_logprob(raw_lines), 20))

    @staticmethod
    def _is_on_topic(cand: Dict[str, Any]) -> bool:
        sc = cand.get("scores", {})
        has_keyword = sc.get("keyword_in_poem", False) or not cand.get("keyword_forms")
        return bool(has_keyword and sc.get("topic_line_fraction", 0.0) >= 0.5)

    @torch.no_grad()
    def score_poem_candidate(
        self,
        candidate: Dict[str, Any],
        keyword: str,
        thematic_field: Dict[str, float],
    ) -> float:
        raw_lines = candidate["raw_lines"]
        flat = [w for l in raw_lines for w in l]
        if not flat:
            return -999.0
        W = self.SCORE_WEIGHTS

        avg_log_p = self._poem_logprob(raw_lines)

        theme_score = sum(thematic_field.get(w, 0.0) for w in flat) / len(flat)

        forms = candidate.get("keyword_forms") or set()
        keyword_hit = 1.0 if any(w in forms for w in flat) else 0.0
        topic_lines = sum(
            1 for l in raw_lines if any(self._on_topic(w, thematic_field) for w in l)
        ) / len(raw_lines)

        rhyme_scheme = candidate.get("rhyme_scheme", "AABB")
        rhyme_eval = self.rhyme_ctrl.score_poem_rhyme_cadence(
            raw_lines,
            scheme=rhyme_scheme,
            w2v=self.w2v,
            line_info=candidate.get("line_info"),
        )
        rhyme_score = rhyme_eval["rhyme_score"]
        # free verse: a chance rhyme must not be rewarded
        rhyme_w = 0.0 if candidate.get("rhyme_scheme") == "none" else W["rhyme"]
        coherence = self._coherence(raw_lines)

        counts = Counter(flat)
        pronoun_stems = candidate.get("pronoun_stems") or set()
        rep_penalty = sum((c - 1) for w, c in counts.items()
                          if c > 1 and w not in FUNCTION_WORDS)
        rep_penalty += sum(
            self.PRONOUN_KEYWORD_PENALTY_SCALE * (c - self.PRONOUN_KEYWORD_FREE_USES)
            for w, c in counts.items()
            if w in FUNCTION_WORDS and self._stem(w) in pronoun_stems
            and c > self.PRONOUN_KEYWORD_FREE_USES
        )

        lengths = [len(l) for l in raw_lines]
        mean_len = sum(lengths) / len(lengths)
        len_std = math.sqrt(sum((x - mean_len) ** 2 for x in lengths) / len(lengths))

        info = candidate.get("line_info") or []
        incomplete = (sum(1 for i in info if not i.get("ended")) / len(info)) if info else 0.0

        composite = (
            W["fluency"] * avg_log_p
            + W["theme"] * theme_score
            + rhyme_w * rhyme_score
            + W["coherence"] * coherence
            + W["keyword"] * keyword_hit
            + W["topic_lines"] * topic_lines
            - W["repetition"] * rep_penalty
            - W["length"] * len_std
            - W["incomplete"] * incomplete
        )

        candidate["scores"] = {
            "total_score": round(composite, 3),
            "fluency_log_prob": round(avg_log_p, 3),
            "thematic_relevance": round(theme_score, 4),
            "keyword_in_poem": bool(keyword_hit),
            "topic_line_fraction": round(topic_lines, 3),
            "rhyme_score": rhyme_score,
            "rhyme_bonus": round(rhyme_w * rhyme_score, 2),
            "coherence": round(coherence, 3),
            "repetition_penalty": round(W["repetition"] * rep_penalty, 2),
            "length_penalty": round(W["length"] * len_std, 2),
            "incomplete_fraction": round(incomplete, 3),
            "rhyme_details": rhyme_eval["details"],
        }
        return composite

    def generate_poem(
        self,
        keyword: str,
        num_lines: int = 4,
        num_candidates: int = 5,
        min_words_per_line: int = 5,
        max_words_per_line: int = 8,
        top_k: int = 20,
        temperature: float = 0.85,
        top_p: float = 0.90,
        use_beam_search: bool = False,
        use_enhancer: bool = True,
        end_mode: Optional[str] = None,
        save_html: bool = False,
        carry_context: bool = True,
        max_perplexity: Optional[float] = None,
        max_rounds: int = 4,
        require_topic: bool = True,
        rhyme_scheme: str = "auto",
        **kwargs
    ) -> Dict[str, Any]:
        """
        rhyme_scheme:
            "AABB" -> force couplet rhyme on every pair (classic scheme)
            "auto" -> rhyme only when the previous line is complete (default)
            "none" -> free verse, no rhyme constraint
        """
        if rhyme_scheme not in VALID_RHYME_SCHEMES:
            rhyme_scheme = "auto"

        if end_mode in ("model", "heuristic"):
            if end_mode == "model" and self.end_id is None:
                raise ValueError("end_mode='model' needs an END token in the vocabulary")
            self.end_mode = end_mode

        num_lines = max(2, min(num_lines, 6))
        keyword_input = keyword
        topic = self._prepare_topic(keyword)
        keyword = topic["keyword"]
        warnings: List[str] = list(topic["warnings"])

        candidates = []
        eligible = []
        rounds = 1 if max_perplexity is None else max(1, max_rounds)
        for _ in range(rounds):
            for _ in range(max(1, num_candidates)):
                cand = self.generate_single_poem(
                    keyword=keyword, num_lines=num_lines,
                    min_words_per_line=min_words_per_line,
                    max_words_per_line=max_words_per_line,
                    top_k=top_k, top_p=top_p, temperature=temperature,
                    use_beam_search=use_beam_search, topic=topic,
                    carry_context=carry_context, rhyme_scheme=rhyme_scheme,
                )
                score = self.score_poem_candidate(cand, keyword, cand["thematic_field"])
                cand["avg_perplexity"] = self._avg_perplexity(cand["raw_lines"])
                candidates.append((score, cand))
            if max_perplexity is None:
                break
            eligible = [
                (s_, c) for s_, c in candidates
                if c["avg_perplexity"] <= max_perplexity
                and (not require_topic or self._is_on_topic(c))
            ]
            if eligible:
                break

        perplexity_reached = None
        pool = candidates
        if max_perplexity is not None:
            perplexity_reached = bool(eligible)
            if eligible:
                pool = eligible
            else:
                pool = sorted(candidates, key=lambda x: x[1]["avg_perplexity"])
                pool = pool[:max(1, len(pool) // 4)]
                warnings.append(
                    f"{len(candidates)}টি চেষ্টায় কোনো কবিতা Perplexity ≤ {max_perplexity} পায়নি; "
                    f"সবচেয়ে কাছেরটি দেখানো হচ্ছে।")

        pool.sort(key=lambda x: x[0], reverse=True)
        best_score, best = pool[0]
        thematic_field = best["thematic_field"]

        enhancement_info = None
        if use_enhancer:
            try:
                enhancer = PoeticEnhancer(vocab=self.vocab, rhyme_controller=self.rhyme_ctrl,
                                          w2v=self.w2v, target_syllables=(18, 26),
                                          function_words=FUNCTION_WORDS)
                enh = enhancer.enhance(best["raw_lines"], keyword=keyword,
                                       scheme=rhyme_scheme if rhyme_scheme != "none" else "AABB")
                if enh["raw_lines"] != best["raw_lines"]:
                    best["raw_lines"] = enh["raw_lines"]
                    best["formatted_lines"] = self._format_lines(best["raw_lines"])
                    best["poem_text"] = "\n".join(best["formatted_lines"])
                    self.score_poem_candidate(best, keyword, thematic_field)
                enhancement_info = {
                    "rhyme_matches": enh["rhyme_matches"],
                    "syllables": enh["syllables"],
                    "steps_applied": enh["steps_applied"],
                    "duplicate_lines": enh["duplicate_lines"],
                    "line_issues": enh["line_issues"],
                    "thematic_overlap": enh["thematic_overlap"],
                }
            except Exception as e:
                import traceback
                print(f"[enhancer] failed: {e}")
                traceback.print_exc()

        raw_lines = best["raw_lines"]
        flat_words = [w for line in raw_lines for w in line]
        total_words = len(flat_words)
        unique_words = len(set(flat_words))
        ttr = round(unique_words / total_words, 4) if total_words else 0.0

        line_pps = [round(math.exp(min(-self._line_logprob(l), 20)), 2) for l in raw_lines]
        avg_pp = round(self._avg_perplexity(raw_lines), 2)      # same definition as max_perplexity
        scores = best["scores"]
        sim_pairs = []
        if self.w2v and getattr(self.w2v, "model", None):
            try:
                sim_pairs = self.w2v.get_similar_words(keyword, topn=8)
            except Exception:
                sim_pairs = []

        lines_on_topic = sum(
            1 for l in raw_lines if any(self._on_topic(w, thematic_field) for w in l))
        if scores.get("topic_line_fraction", 0.0) < 0.5:
            warnings.append("কবিতাটি বিষয়ের সাথে কম মিলেছে; আবার চেষ্টা করুন বা Temperature কমিয়ে দেখুন।")

        result = {
            "keyword": keyword,
            "keyword_input": keyword_input,
            "num_lines": len(raw_lines),
            "lines": best["formatted_lines"],
            "poem_text": best["poem_text"],
            "token_lines": raw_lines,
            "word2vec_similar_words": sim_pairs,
            "theme_seeds_used": best["theme_words"][:10],
            "thematic_coverage": best["thematic_coverage"],
            "rhyme_scheme": rhyme_scheme,
            "topic": {
                "keyword": keyword,
                "keyword_forms": sorted(topic["forms"]),
                "field_words": best["theme_words"][:12],
                "lines_on_topic": lines_on_topic,
                "keyword_in_poem": scores.get("keyword_in_poem", False),
            },
            "warnings": warnings,
            "scoring_details": scores,
            "candidates_evaluated": len(candidates),
            "perplexity_target": max_perplexity,
            "perplexity_reached": perplexity_reached,
            "end_mode": self.end_mode,
            "line_info": best["line_info"],
            "debug_trace": None,
            "enhancement": enhancement_info,
            "metrics": {
                "total_words": total_words,
                "unique_words": unique_words,
                "type_token_ratio": ttr,
                "line_perplexities": line_pps,
                "average_perplexity": avg_pp,
                "rhyme_cadence_score": scores["rhyme_score"],
                "rhyme_scheme": rhyme_scheme,
                "average_words_per_line": round(total_words / max(len(raw_lines), 1), 2),
                "thematic_coverage": best["thematic_coverage"],
                "topic_line_fraction": scores.get("topic_line_fraction", 0.0),
                "keyword_in_poem": scores.get("keyword_in_poem", False),
                "coherence": scores.get("coherence", 0.0),
            },
        }

        if save_html:
            try:
                from src.utils import save_poem_to_html
                save_poem_to_html(
                    best["formatted_lines"], keyword=keyword,
                    metadata={"total_words": total_words, "ttr": ttr,
                              "avg_perplexity": avg_pp, "rhyme_score": scores["rhyme_score"]},
                    rhyme_scheme=rhyme_scheme)
            except Exception as e:
                print(f"[save_html] skipped: {e}")

        return result