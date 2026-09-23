import re
import unicodedata
from typing import List, Dict, Tuple, Optional, Set

from src.rhyme_controller import BengaliRhymeController



KNOWN_JUNK_TOKENS = {
    "সমুদ্দুরসমুদ্র", "সমুদ্রসমুদ্র", "সমুদ্দুরসমুদ্দুর",
    "কিমও", "কাৎরা", "খটকার", "পানুই",
    "ঝরে—ঘাসে", "ঝরে—রোদে", "ঝরে—মধুর",
}

ALLOWED_SINGLE_CHARS = {'এ', 'ও', 'ঐ'}
_NON_BENGALI = re.compile(r'[^\u0980-\u09FF]')

PRONOUN_VERB_SUFFIXES: Dict[str, Tuple[str, ...]] = {
    'আমি': ('ি', 'ছি', 'লাম', 'তাম', 'ব', 'বো'),
    'তুমি': ('ো', 'ছ', 'ছো', 'লে', 'বে'),
    'তোমরা': ('ো', 'ছ', 'ছো', 'লে', 'বে'),
    'সে': ('ে', 'ছে', 'ল', 'লেন', 'বে', 'বেন'),
    'তিনি': ('েন', 'ছেন', 'লেন', 'বেন'),
}
# NEW
ALL_PRONOUNS = {
    'আমি', 'আমরা', 'তুমি', 'তোমরা', 'সে', 'তারা', 'এরা', 'ওরা',
    'তিনি', 'তাঁরা', 'উনি', 'ইনি', 'এ', 'ও', 'ঐ', 'ওই', 'সেই',
    'তা', 'এটা', 'ওটা', 'সেটা', 'এটি', 'ওটি', 'সেটি',
}

IMPERATIVE_ENDINGS = ('ো', 'োও')


class BengaliSyllableCounter:
    VOWEL_SIGNS = set('ািীুূৃেৈোৌ')
    INDEPENDENT_VOWELS = set('অআইঈউঊঋএঐওঔ')
    CONSONANTS = set('কখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহড়ঢ়য়ৎংঃঁ')

    @classmethod
    def count(cls, word: str) -> int:
        if not word:
            return 0
        word = unicodedata.normalize('NFC', word)
        syllables = 0
        i = 0
        n = len(word)

        while i < n:
            ch = word[i]
            if ch in cls.INDEPENDENT_VOWELS:
                syllables += 1
                i += 1
                while i < n and word[i] in cls.VOWEL_SIGNS:
                    i += 1
                continue
            if ch in cls.CONSONANTS:
                syllables += 1
                i += 1
                while i < n and word[i] in cls.VOWEL_SIGNS:
                    i += 1
                if i < n and word[i] == '\u09CD':
                    i += 1
                    if i < n and word[i] in cls.CONSONANTS:
                        syllables -= 1
                continue
            i += 1

        return max(syllables, 1)

    @classmethod
    def count_line(cls, tokens: List[str]) -> int:
        return sum(cls.count(t) for t in tokens)


class BengaliGrammarHeuristics:
    CONSECUTIVE_VERB_SUFFIXES = (
        "ল", "লো", "লে", "ছি", "ছো", "ছে", "ব", "বে",
        "তাম", "তেন", "বেন", "গেল", "গেছে",
    )

    COMMON_POSTPOSITIONS = {
        "থেকে", "দিয়ে", "জন্য", "সাথে", "মধ্যে", "উপরে",
        "নিচে", "পরে", "আগে", "কাছে", "দিকে", "মতো",
        "সহ", "বিনা", "ছাড়া",
    }

    BAD_PATTERNS = {
        "সমুদ্দুরসমুদ্র", "সমুদ্রসমুদ্র", "কিমও", "কাৎরা",
        "খটকার", "পানুই",
    }

    @classmethod
    def has_consecutive_verbs(cls, tokens: List[str]) -> bool:
        if len(tokens) < 2:
            return False
        verbs = 0
        for t in tokens:
            # len(t) >= 3 avoids treating short nouns (জল, সব, ফুল) as verbs
            if len(t) >= 3 and any(t.endswith(s) for s in cls.CONSECUTIVE_VERB_SUFFIXES):
                verbs += 1
                if verbs >= 3:
                    return True
            else:
                verbs = 0
        return False

    @classmethod
    def has_repeated_word(cls, tokens: List[str]) -> bool:
        for i in range(len(tokens) - 1):
            if tokens[i] == tokens[i + 1]:
                return True
        return False

    @classmethod
    def has_junk(cls, tokens: List[str]) -> bool:
        for t in tokens:
            if t in cls.BAD_PATTERNS:
                return True
            if cls._looks_like_concatenation(t):
                return True
        return False

    @staticmethod
    def _looks_like_concatenation(tok: str) -> bool:
        if len(tok) < 8:
            return False
        for L in range(4, len(tok) // 2 + 1):
            for start in range(len(tok) - 2 * L + 1):
                sub = tok[start:start + L]
                if tok.count(sub) >= 2:
                    return True
        return False

    @classmethod
    def has_isolated_single_char(cls, tokens: List[str]) -> bool:
        for t in tokens[1:-1]:
            if len(t) == 1 and t not in ALLOWED_SINGLE_CHARS:
                return True
        return False

    @classmethod
    def has_disconnected_ending(cls, tokens: List[str], w2v=None, min_sim: float = 0.12) -> bool:

        if w2v is None or getattr(w2v, "model", None) is None:
            return False
        if len(tokens) < 3:
            return False
        last = tokens[-1]
        rest = [t for t in tokens[:-1] if len(t) >= 2]
        if not rest:
            return False
        sims = []
        for t in rest:
            try:
                s = w2v.get_similarity(last, t)
            except Exception:
                s = 0.0
            if s > 0:
                sims.append(s)
        if not sims:
            return False
        return (sum(sims) / len(sims)) < min_sim

    @classmethod
    def has_person_mismatch(cls, tokens: List[str]) -> bool:

        present_pronouns = [t for t in tokens if t in PRONOUN_VERB_SUFFIXES]
        if not present_pronouns:
            return False
        for pronoun in present_pronouns:
            my_suffixes = PRONOUN_VERB_SUFFIXES[pronoun]
            other_suffixes: Set[str] = set()
            for p, sufs in PRONOUN_VERB_SUFFIXES.items():
                if p != pronoun:
                    other_suffixes.update(sufs)
            other_only = other_suffixes - set(my_suffixes)
            for t in tokens:
                if t == pronoun or t in PRONOUN_VERB_SUFFIXES or len(t) < 3:
                    continue
                matches_other = any(t.endswith(s) for s in other_only)
                matches_mine = any(t.endswith(s) for s in my_suffixes)
                if matches_other and not matches_mine:
                    return True
        return False

    @classmethod
    def has_negation_mismatch(cls, tokens: List[str]) -> bool:
        for i in range(len(tokens) - 1):
            t, nxt = tokens[i], tokens[i + 1]
            if nxt == 'নি' and len(t) >= 2 and any(t.endswith(s) for s in IMPERATIVE_ENDINGS):
                return True
        return False
    
    @classmethod
    def has_pronoun_clash(cls, tokens: List[str]) -> bool:
        for i in range(len(tokens) - 1):
            a, b = tokens[i], tokens[i + 1]
            if a in ALL_PRONOUNS and b in ALL_PRONOUNS and a != b:
                return True
        return False

    @classmethod
    def has_bare_pronoun_ending(cls, tokens: List[str]) -> bool:
        if len(tokens) < 3:
            return False
        last = tokens[-1]
        if last not in ALL_PRONOUNS:
            return False
        verb_suffixes = ('ি', 'ছি', 'লাম', 'তাম', 'ব', 'বো', 'ো', 'ছ', 'ছো',
                          'লে', 'বে', 'ে', 'ছে', 'ল', 'লেন', 'েন', 'ছেন', 'বেন')
        has_verb = any(
           len(t) >= 3 and any(t.endswith(s) for s in cls.CONSECUTIVE_VERB_SUFFIXES)
            for t in tokens[:-1]
        )
        return not has_verb

    @classmethod
    def has_pronoun_flood(
        cls, tokens: List[str], function_words: Optional[Set[str]] = None, max_fraction: float = 0.65
    ) -> bool:
        if not function_words or not tokens:
            return False
        func_count = sum(1 for t in tokens if t in function_words)
        return (func_count / len(tokens)) > max_fraction

    @classmethod
    def is_grammatical(
        cls,
        tokens: List[str],
        w2v=None,
        function_words: Optional[Set[str]] = None,
    ) -> bool:
        if len(tokens) < 3:
            return False
        if cls.has_consecutive_verbs(tokens):
            return False
        if cls.has_repeated_word(tokens):
            return False
        if cls.has_junk(tokens):
            return False
        if cls.has_isolated_single_char(tokens):
            return False
        if cls.has_disconnected_ending(tokens, w2v=w2v):
            return False
        if cls.has_person_mismatch(tokens):
            return False
        if cls.has_negation_mismatch(tokens):
            return False
        if cls.has_pronoun_clash(tokens):
            return False
        if cls.has_bare_pronoun_ending(tokens):
            return False
        if cls.has_pronoun_flood(tokens, function_words=function_words):
            return False
        has_content = any(len(t) >= 3 for t in tokens)
        if not has_content:
            return False
        return True


def is_bad_token(word: str) -> bool:

    if not word:
        return True
    if word in KNOWN_JUNK_TOKENS or word in BengaliGrammarHeuristics.BAD_PATTERNS:
        return True
    if len(word) > 30:
        return True
    if len(word) == 1 and word not in ALLOWED_SINGLE_CHARS:
        return True
    if word.startswith('\u09DF'):
        return True
    # tokens from an old, badly tokenised vocab ('পাখি,আকাশ', latin letters, digits ...)
    if _NON_BENGALI.search(word):
        return True
    if len(word) >= 6 and len(word) % 2 == 0:
        half = len(word) // 2
        if word[:half] == word[half:]:
            return True
    if len(word) >= 6:
        for l in (2, 3):
            if len(word) % l == 0 and word[:len(word) // l] * l == word:
                return True
    if BengaliGrammarHeuristics._looks_like_concatenation(word):
        return True
    return False


class PoeticEnhancer:

    def __init__(
        self,
        vocab,
        rhyme_controller: BengaliRhymeController,
        w2v=None,
        target_syllables: Optional[Tuple[int, int]] = (18, 26),
        function_words: Optional[Set[str]] = None,
    ):
        self.vocab = vocab
        self.rhyme_ctrl = rhyme_controller
        self.w2v = w2v
        self.target_syllables = target_syllables
        self.function_words = function_words or set()

    def clean_concatenations(self, raw_lines: List[List[str]]) -> List[List[str]]:
        cleaned_lines = []
        for line in raw_lines:
            cleaned = []
            for tok in line:
                if tok in KNOWN_JUNK_TOKENS:
                    continue
                if BengaliGrammarHeuristics._looks_like_concatenation(tok):
                    continue
                if len(tok) <= 2 and tok not in self.vocab.word2idx:
                    continue
                cleaned.append(tok)
            cleaned_lines.append(cleaned if cleaned else list(line))
        return cleaned_lines

    def find_duplicate_lines(self, lines: List[List[str]]) -> List[int]:
        dups, seen = [], set()
        for i, line in enumerate(lines):
            sig = tuple(line[:3])
            if sig in seen:
                dups.append(i)
            seen.add(sig)
        return dups

    def count_rhyme_matches(self, lines: List[List[str]]) -> int:
        matches = 0
        for i in range(0, len(lines) - 1, 2):
            a, b = lines[i], lines[i + 1]
            if a and b and self.rhyme_ctrl.is_rhyme(a[-1], b[-1]):
                matches += 1
        return matches

    def thematic_overlap(self, lines: List[List[str]], keyword: str) -> List[int]:
        thematic: Set[str] = {keyword}
        if self.w2v and getattr(self.w2v, "model", None):
            try:
                for w, _ in self.w2v.get_similar_words(keyword, topn=15):
                    thematic.add(w)
            except Exception:
                pass
        return [sum(1 for t in line if t in thematic) for line in lines]

    def line_issues(self, lines: List[List[str]]) -> List[List[str]]:
        lo_hi = self.target_syllables
        out = []
        for line in lines:
            tags = []
            if not BengaliGrammarHeuristics.is_grammatical(
                line, w2v=self.w2v, function_words=self.function_words
            ):
                tags.append("ungrammatical_heuristic")
            if lo_hi:
                syl = BengaliSyllableCounter.count_line(line)
                if syl < lo_hi[0]:
                    tags.append("short")
                elif syl > lo_hi[1]:
                    tags.append("long")
            out.append(tags)
        return out

    def filter_ungrammatical_lines(self, raw_lines, min_lines: int = 2):
        filtered = [
            l for l in raw_lines
            if BengaliGrammarHeuristics.is_grammatical(
                l, w2v=self.w2v, function_words=self.function_words
            )
        ]
        return filtered if len(filtered) >= min_lines else raw_lines

    def dedupe_lines(self, raw_lines):
        keep, seen = [], set()
        for line in raw_lines:
            sig = tuple(line[:3]) if line else ()
            if sig in seen:
                continue
            seen.add(sig)
            keep.append(line)
        return keep

    def enforce_rhyme(self, raw_lines, scheme: str = "AABB"):
        """Deprecated: kept for API compatibility. Reports matches; NEVER swaps words."""
        return [list(l) for l in raw_lines], self.count_rhyme_matches(raw_lines)

    def enhance(
        self,
        raw_lines: List[List[str]],
        keyword: str,
        scheme: str = "AABB",
    ) -> Dict:
        applied: List[str] = []

        lines = self.clean_concatenations(raw_lines)
        applied.append("concat_cleanup")

        duplicates = self.find_duplicate_lines(lines)
        applied.append("duplicate_check")

        issues = self.line_issues(lines)
        applied.append("grammar_and_length_check")

        overlap = self.thematic_overlap(lines, keyword)
        applied.append("thematic_overlap")

        matches = self.count_rhyme_matches(lines)
        applied.append("rhyme_check")

        syllables = [BengaliSyllableCounter.count_line(l) for l in lines]

        return {
            "raw_lines": lines,
            "syllables": syllables,
            "rhyme_matches": matches,
            "steps_applied": applied,
            "duplicate_lines": duplicates,
            "line_issues": issues,
            "thematic_overlap": overlap,
        }


def count_syllables(word: str) -> int:
    return BengaliSyllableCounter.count(word)


def is_grammatical_line(
    tokens: List[str], w2v=None, function_words: Optional[Set[str]] = None
) -> bool:
    return BengaliGrammarHeuristics.is_grammatical(tokens, w2v=w2v, function_words=function_words)