import re
import unicodedata
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional, Set, Any


class BengaliRhymeController:

    POETIC_CODAS = [
        'ায়', 'ার', 'ান', 'াণ', 'াল', 'াশ', 'াষ', 'াস', 'াত',
        'ুর', 'িল', 'েশ', 'েষ', 'েস', 'োর', 'েন', 'ের', 'িত', 'ীয়',
        'িয়', 'য়'
    ]

    INDEP_TO_SIGN = {
        'আ': 'া', 'ই': 'ি', 'ঈ': 'ি', 'উ': 'ু',
        'ঊ': 'ু', 'ঋ': 'ৃ', 'এ': 'ে', 'ঐ': 'ৈ',
        'ও': 'ো', 'ঔ': 'ৌ',
    }

    VOWEL_SIGNS = 'ািীুূৃেৈোৌ'

    def __init__(self, line_end_counts: Optional[Counter] = None, w2v: Optional[Any] = None):
        self.rhyme_families: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        self.valid_endings: Set[str] = set()
        self.end_word_counts: Counter = Counter()
        self.w2v = w2v

        if line_end_counts:
            self.index_corpus_line_endings(line_end_counts)

    @staticmethod
    def phonetic_simplify(word: str) -> str:
        if not word or not isinstance(word, str):
            return ""
        w = unicodedata.normalize('NFC', word)
        w = w.replace('\u09AF\u09BC', '\u09DF')
        w = w.replace('\u09A1\u09BC', '\u09DC')
        w = w.replace('\u09A2\u09BC', '\u09DD')
        w = w.replace('ী', 'ি').replace('ূ', 'ু')
        w = w.replace('ণ', 'ন')
        w = w.replace('ষ', 'শ').replace('স', 'শ')
        w = w.replace('ঢ়', 'ড়')
        return w

    @classmethod
    def extract_rhyme_key(cls, word: str) -> str:

        if not word or not isinstance(word, str):
            return ""
        clean_w = re.sub(r'^[^\u0980-\u09FF]+|[^\u0980-\u09FF]+$', '', word)
        if len(clean_w) < 2:
            return clean_w

        w_phon = cls.phonetic_simplify(clean_w)

        # 1. Match against known Bengali poetic multi-char codas
        for coda in cls.POETIC_CODAS:
            coda_phon = cls.phonetic_simplify(coda)
            if w_phon.endswith(coda_phon):
                return coda_phon

        # 2. Match terminal syllable: Consonant + Matra
        if w_phon[-1] in cls.VOWEL_SIGNS:
            if len(w_phon) >= 2:
                return w_phon[-2:]
            return w_phon[-1]

        # 3. FIXED: consonant-final — keep vowel of last syllable
        if len(w_phon) < 2:
            return w_phon

        if len(w_phon) >= 3 and w_phon[-2] == '\u09CD':
            # Conjunct coda: ধর্ম -> 'অর্ম'
            return 'অ' + w_phon[-3:]

        prev = w_phon[-2]

        # If previous char is a vowel sign
        if prev in cls.VOWEL_SIGNS:
            return prev + w_phon[-1]

        # If previous char is an independent vowel
        if prev in cls.INDEP_TO_SIGN:
            return cls.INDEP_TO_SIGN[prev] + w_phon[-1]

        # Default: inherent vowel /ɔ/
        return 'অ' + w_phon[-1]

    @classmethod
    def is_rhyme(cls, w1: str, w2: str) -> bool:
        if not w1 or not w2 or w1 == w2:
            return False
        k1 = cls.extract_rhyme_key(w1)
        k2 = cls.extract_rhyme_key(w2)
        if not k1 or not k2:
            return False
        return k1 == k2

    @classmethod
    def rhyme_similarity(cls, w1: str, w2: str) -> float:
        if not w1 or not w2 or w1 == w2:
            return 0.0
        k1 = cls.extract_rhyme_key(w1)
        k2 = cls.extract_rhyme_key(w2)
        if k1 and k2 and k1 == k2:
            return 1.0
        if w1[-1] == w2[-1] and w1[-1] in 'ািীুূেোৌ':
            return 0.5
        return 0.0

    def index_corpus_line_endings(self, line_end_counts: Counter, min_freq: int = 2):
        self.rhyme_families.clear()
        self.valid_endings.clear()
        self.end_word_counts = line_end_counts.copy()

        for word, cnt in line_end_counts.items():
            if cnt < min_freq:
                continue
            if len(word) < 2:
                continue
            r_key = self.extract_rhyme_key(word)
            if r_key:
                self.rhyme_families[r_key].append((word, cnt))
                self.valid_endings.add(word)

        for r_key in self.rhyme_families:
            self.rhyme_families[r_key].sort(key=lambda x: x[1], reverse=True)

    def get_rhyming_words(
        self,
        target_word: str,
        top_n: int = 12,
        theme_words: Optional[Set[str]] = None,
        w2v=None
    ) -> List[str]:
        r_key = self.extract_rhyme_key(target_word)
        if not r_key or r_key not in self.rhyme_families:
            return []

        candidates = [w for w, _ in self.rhyme_families[r_key] if w != target_word]
        if not candidates:
            return []

        scored = []
        for cand in candidates:
            freq_score = min(1.0, self.end_word_counts.get(cand, 1) / 30.0)
            theme_score = 1.0 if (theme_words and cand in theme_words) else 0.0
            sem_score = 0.0
            if w2v:
                sem_score = max(0.0, w2v.get_similarity(target_word, cand))
            composite = 0.4 * freq_score + 0.4 * theme_score + 0.2 * sem_score
            scored.append((composite, cand))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [w for _, w in scored[:top_n]]

    def plan_aabb_couplets(
        self,
        keyword: str,
        theme_words: List[str],
        w2v=None,
        num_lines: int = 4
    ) -> List[Tuple[Optional[str], Optional[str]]]:
        theme_set = set(theme_words + [keyword])
        pairs = []
        used_words = set()

        def find_couplet(preferred_seeds: List[str]) -> Tuple[Optional[str], Optional[str]]:
            for seed in preferred_seeds:
                if seed in used_words or seed not in self.valid_endings:
                    continue
                rhymes = self.get_rhyming_words(seed, top_n=10, theme_words=theme_set, w2v=w2v)
                available_rhymes = [r for r in rhymes if r not in used_words]
                if available_rhymes:
                    e1 = seed
                    e2 = available_rhymes[0]
                    used_words.add(e1)
                    used_words.add(e2)
                    return e1, e2

            rich_keys = ['রা', 'ান', 'াশ', 'ল', 'দি', '়ে', 'োর', 'াত', 'িল']
            for k in rich_keys:
                if k in self.rhyme_families and len(self.rhyme_families[k]) >= 2:
                    words = [w for w, _ in self.rhyme_families[k] if w not in used_words]
                    if len(words) >= 2:
                        e1, e2 = words[0], words[1]
                        used_words.add(e1)
                        used_words.add(e2)
                        return e1, e2

            return None, None

        seed_list_1 = [keyword] + theme_words
        c1_w1, c1_w2 = find_couplet(seed_list_1)
        pairs.append((c1_w1, c1_w2))

        poetic_seeds = [w for w in theme_words if w not in used_words] + [
            'গান', 'প্রাণ', 'আকাশ', 'বাতাস', 'বেলা', 'খেলা', 'রাত', 'প্রভাত',
            'আলো', 'ভালো', 'দূর', 'সুর', 'জল', 'তল', 'ঝিল', 'নীল'
        ]
        c2_w1, c2_w2 = find_couplet(poetic_seeds)
        pairs.append((c2_w1, c2_w2))

        if num_lines >= 6:
            c3_w1, c3_w2 = find_couplet(poetic_seeds)
            pairs.append((c3_w1, c3_w2))

        return pairs

    def score_poem_rhyme_cadence(
        self,
        raw_lines: List[List[str]],
        scheme: str = 'AABB',
        w2v: Optional[Any] = None,
        line_info: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        if len(raw_lines) < 2:
            return {"rhyme_score": 0.0, "scheme": scheme, "rhyme_pairs": [], "details": []}

        end_words = [line[-1] if line else "" for line in raw_lines]
        matches = 0
        total_checks = 0
        pair_details = []
        rhyme_pairs = []

        if scheme in ('AABB', 'auto'):
            for i in range(0, len(end_words) - 1, 2):
                w1 = end_words[i]
                w2 = end_words[i + 1]

                # When line_info is provided, only count couplets where target_rhyme was actually set.
                # In 'auto' scheme, intentional non-rhyme lines should not be penalized.
                if line_info and (i + 1) < len(line_info):
                    target = line_info[i + 1].get("target_rhyme")
                    if not target:
                        pair_details.append({
                            "lines": (i + 1, i + 2),
                            "words": (w1, w2),
                            "coda1": self.extract_rhyme_key(w1),
                            "coda2": self.extract_rhyme_key(w2),
                            "is_rhyme": False,
                            "target_rhyme": None,
                            "skipped": True
                        })
                        continue

                total_checks += 1
                sim = self.rhyme_similarity(w1, w2)
                is_rhyme = sim >= 0.7
                if is_rhyme:
                    matches += 1
                    rhyme_pairs.append((w1, w2))
                pair_details.append({
                    "lines": (i + 1, i + 2),
                    "words": (w1, w2),
                    "coda1": self.extract_rhyme_key(w1),
                    "coda2": self.extract_rhyme_key(w2),
                    "is_rhyme": is_rhyme
                })

        score = (matches / total_checks) if total_checks > 0 else 0.0

        # Semantic check: for each rhyming pair, check semantic similarity of the two end words
        w2v_model = w2v or getattr(self, "w2v", None)
        if w2v_model and rhyme_pairs:
            for pair in rhyme_pairs:
                try:
                    sem = w2v_model.get_similarity(pair[0], pair[1])
                except Exception:
                    sem = 0.0
                if sem < 0.15:      # very different semantic words rhyming is suspicious
                    score *= 0.7

        return {
            "rhyme_score": round(score, 3),
            "scheme": scheme,
            "pairs_matched": matches,
            "total_pairs": total_checks,
            "details": pair_details
        }