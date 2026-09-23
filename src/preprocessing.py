import os
import re
import random
import pickle
import hashlib
import unicodedata
import pandas as pd
from collections import Counter
from typing import List, Dict, Tuple, Optional, Any


BENGALI_STOP_WORDS = {
    'অতএব', 'অথবা', 'অনুরূপ', 'অনেক', 'অন্য', 'অবশ্য', 'অবিলম্বে', 'অর্থাত',
    'আইন', 'আগে', 'আজ', 'আজেবাজে', 'আবার', 'আমরা', 'আমাদের', 'আমাকে',
    'আমার', 'আমি', 'আর', 'আরও', 'ইত্যাদি', 'ইহা', 'উচিত', 'উত্তর', 'উপরে',
    'উভয়', 'একই', 'একটি', 'একবার', 'একসেট', 'একে', 'এখন', 'এখানে', 'এটাই',
    'এটা', 'এটি', 'এত', 'এতে', 'এমন', 'এমনকি', 'এর', 'এরা', 'এল',
    'এস', 'এসে', 'ঐ', 'ও', 'ওঁদের', 'ওঁর', 'ওঁরা', 'ওই', 'ওকে', 'ওদের',
    'ওর', 'ওরা', 'কখনও', 'কত', 'কবে', 'কমনে', 'কয়েক', 'কয়েকটি', 'করে',
    'করেই', 'করেছিল', 'করছেন', 'করতে', 'করবে', 'করলেন', 'করা',
    'করাই', 'করায়', 'করার', 'করি', 'করিতে', 'করিয়া', 'করিয়ে',
    'করেছেন', 'কাউকে', 'কাছ', 'কাছে', 'কাজ', 'কাজে', 'কারও', 'কারণ', 'কী',
    'কুটি', 'কে', 'কেউ', 'কেন', 'কেমন', 'কোটি', 'কোন', 'কোনও', 'কোনো',
    'ক্ষেত্রে', 'খুব', 'খুঁজে', 'গিয়ে', 'গেল', 'গেলে', 'গেছে', 'চাওয়া',
    'চাওয়াটা', 'চাই', 'চায়', 'চার', 'চালু', 'চেয়ে', 'চেষ্টা', 'ছাড়া',
    'ছোট', 'জন', 'জনকে', 'জনের', 'জন্য', 'জন্যও', 'জরুরি', 'জানতে', 'জানা',
    'জানানো', 'জানায়', 'জানিয়ে', 'জানিয়েছে', 'জেলে', 'জোড়া', 'ঠিক',
    'তখন', 'তত', 'তথা', 'তবু', 'তবে', 'তা', 'তাঁকে', 'তাঁদের', 'তাঁর',
    'তাঁরা', 'তাঁহাই', 'তাই', 'তাও', 'তাকে', 'তাদের', 'তার', 'তারই', 'তাহলে',
    'তাহা', 'তাহাতে', 'তাহার', 'তিন', 'তিনি', 'তিনিও', 'তুলে', 'তুমি', 'তো',
    'তোমরা', 'তোমাদের', 'তোমাকে', 'তোমার', 'থাকবে', 'থাকবেন', 'থাকলে', 'থাকে',
    'থাকেন', 'থেকে', 'থেকেই', 'দিয়ে', 'দিয়েছে', 'দিয়েছেন', 'দিলেন', 'দুটি',
    'দুটো', 'দেওয়া', 'দেওয়ার', 'দেওয়াটা', 'দেখে', 'দেখা', 'দেখেছেন', 'দেবেন',
    'দেয়', 'দ্বারা', 'ধরে', 'ধরা', 'নইলে', 'নও', 'না', 'নাকি', 'নাগাদ',
    'নিয়ে', 'নেওয়া', 'নেওয়ার', 'নেই', 'পক্ষে', 'পর', 'পরে', 'পরেই', 'পরেও',
    'পর্যন্ত', 'পাওয়া', 'পারবে', 'পারবেন', 'পারি', 'পারে', 'পারেন', 'পাশের',
    'প্রায়', 'প্রাপ্ত', 'ফলে', 'ফের', 'বলা', 'বলল', 'বললেন', 'বলতে', 'বলে',
    'বলেছিলেন', 'বলেন', 'বসে', 'বহু', 'বা', 'বাদে', 'বার', 'বারবার', 'বেশ',
    'বেশি', 'মোট', 'মোটেই', 'যখন', 'যত', 'যতটা', 'যতটুকু', 'যথা', 'যদিয়',
    'যদি', 'যথেষ্ট', 'যাওয়া', 'যায়', 'যায়ে', 'যার', 'যারা', 'যিনি', 'যে',
    'যেখানে', 'যেহেতু', 'রকম', 'রয়ে', 'রয়েছে', 'রাখা', 'রেখে', 'লাগে',
    'লাগবে', 'শুধু', 'শুরু', 'সঙ্গে', 'সঙ্গেও', 'সব', 'সবাই', 'সবাইকে',
    'সবসময়', 'সহ', 'সরাসরি', 'সহিত', 'সাধারণ', 'সামনে', 'সিঁড়ি',
    'সে', 'সেই', 'সেখান', 'সেখানে', 'সেটা', 'সেটাই', 'সেটি', 'সেরকম',
    'হওয়া', 'হওয়ার', 'হওয়াটা', 'হতে', 'হতেই', 'হবে', 'হবেন', 'হয়ে', 'হয়েছিল',
    'হয়েছেন', 'হয়েও', 'হয়', 'হয়নি', 'হলো', 'হল'
}


class BengaliPreprocessor:

    EXTRA_LABEL_MAP = {
        "প্রেমমূলক":       "Love",
        "চিন্তামূলক":       "Metaphor",
        "মানবতাবাদী":      "Humanity",
        "ভক্তিমূলক":        "Religious",
        "রূপক":            "Metaphor",
        "প্রকৃতিমূলক":      "Nature",
        "ছড়া":             "Children",
        "স্বদেশমূলক":       "Patriotic",
        "নীতিমূলক":         "Policy",
        "শোকমূলক":         "Separation",
        "যুদ্ধমূলক":        "War",
        "সনেট":            "Miscellaneous",
        "হাস্যরসাত্মক":    "Miscellaneous",
        "কাহিনীকাব্য":      "Miscellaneous",
        "গীতিগাথা":         "Miscellaneous",
        "ব্যঙ্গাত্মক":      "Miscellaneous",
        "মহাকাব্য":         "Miscellaneous",
        "গীতিনাট্য":        "Miscellaneous",
        "নাটকীয়":          "Miscellaneous",
        "পত্রকাব্য":        "Miscellaneous",
        "ব্যালাড":          "Miscellaneous",
        "গীতি-ব্যালাড":     "Miscellaneous",
        "মিসলেনিয়াস":       "Miscellaneous",
        "Love Erotic":       "Love",
        "Love":              "Love",
        "Nature":            "Nature",
        "Patriotic":         "Patriotic",
        "Devotional":        "Religious",
        "Religious":         "Religious",
        "Humanism":          "Humanity",
        "Humanity":          "Humanity",
        "Reflective":        "Metaphor",
        "Allegory":          "Metaphor",
        "Philosophical":     "Metaphor",
        "Didactic":          "Policy",
        "Policy":            "Policy",
        "Children":          "Children",
        "Rhymes":            "Children",
        "War":               "War",
        "Elegy":             "Separation",
        "Separation":        "Separation",
        "Sonnet":            "Miscellaneous",
        "Comedy":            "Miscellaneous",
        "Narrative Story":   "Miscellaneous",
        "Lyrical Ballad":    "Miscellaneous",
        "Satire":            "Miscellaneous",
        "Ode":               "Miscellaneous",
        "Epic":              "Miscellaneous",
        "Ballads":           "Miscellaneous",
        "Dramatic":          "Miscellaneous",
        "Epistle":           "Miscellaneous",
        "Lyrical Drama":     "Miscellaneous",
        "Miscellaneous":     "Miscellaneous",
    }

    def __init__(self, data_path: str = "data/SAHITTO.ods"):
        self.data_path = data_path
        self.raw_df: Optional[pd.DataFrame] = None
        self.cleaned_df: Optional[pd.DataFrame] = None
        self.corpus_lines: List[List[str]] = []
        self.corpus_poems: List[List[str]] = []
        self.corpus_labeled_poems: List[Dict[str, Any]] = []
        self.vocab_counts: Counter = Counter()
        self.line_end_counts: Counter = Counter()
        self.line_start_counts: Counter = Counter()
        # poem-level fingerprints, shared by the ODS and the extra-data loader
        self._seen_poem_hashes: set = set()

    def load_and_inspect_dataset(self) -> Dict[str, Any]:
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset file not found at: {self.data_path}")

        print(f"Loading dataset directly from {self.data_path}...")
        self.raw_df = pd.read_excel(self.data_path, engine="odf")

        total_records = len(self.raw_df)
        columns = list(self.raw_df.columns)
        missing_values = self.raw_df.isnull().sum().to_dict()
        total_exact_duplicates = int(self.raw_df.duplicated().sum())
        poem_duplicates = int(self.raw_df['poem'].duplicated().sum()) if 'poem' in self.raw_df else 0

        writer_counts = self.raw_df['writer'].value_counts().head(10).to_dict() if 'writer' in self.raw_df else {}
        label_counts = self.raw_df['label'].value_counts().to_dict() if 'label' in self.raw_df else {}

        sample_entries = []
        for idx, row in self.raw_df.head(3).iterrows():
            sample_entries.append({
                "title": str(row.get('title', '')),
                "writer": str(row.get('writer', '')),
                "label": str(row.get('label', '')),
                "poem_snippet": str(row.get('poem', ''))[:150].replace('\n', ' ') + "..."
            })

        return {
            "file_path": self.data_path,
            "total_records": total_records,
            "columns": columns,
            "text_column": "poem",
            "poet_column": "writer",
            "category_column": "label",
            "missing_values": missing_values,
            "exact_duplicates": total_exact_duplicates,
            "duplicate_poems": poem_duplicates,
            "unique_writers": int(self.raw_df['writer'].nunique()) if 'writer' in self.raw_df else 0,
            "top_writers": writer_counts,
            "unique_labels": int(self.raw_df['label'].nunique()) if 'label' in self.raw_df else 0,
            "label_distribution": label_counts,
            "sample_entries": sample_entries,
            "recommended_text_column": "poem"
        }

    @staticmethod
    def normalize_bengali_unicode(text: str) -> str:
        if not text or not isinstance(text, str):
            return ""
        text = unicodedata.normalize('NFC', text)
        text = text.replace('\u09AF\u09BC', '\u09DF')
        text = text.replace('\u09A1\u09BC', '\u09DC')
        text = text.replace('\u09A2\u09BC', '\u09DD')
        return text

    BENGALI_BASE = r'\u0985-\u09B9\u09DC-\u09DF'
    BENGALI_ALL  = r'\u0980-\u09FF'

    @classmethod
    def is_valid_bengali_word(cls, word: str) -> bool:
        if not word or not isinstance(word, str):
            return False
        word = cls.normalize_bengali_unicode(word)

        if re.match(r'^[\u09BE-\u09CD\u09D7\u0981-\u0983]', word):
            return False
        if word.startswith('\u09DF'):
            return False
        if len(word) == 1 and word not in {'এ', 'ও', 'ঐ'}:
            return False
        if re.fullmatch(r'[\u0995-\u09B9]\u09CD?', word):
            return False
        if not re.search(r'[\u0985-\u09B9\u09DC-\u09DF]', word):
            return False
        if re.search(r'[a-zA-Z0-9]', word):
            return False
        return True

    @classmethod
    def clean_bengali_text(cls, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
        text = re.sub(r'\S+@\S+', ' ', text)
        text = re.sub(r'<.*?>', ' ', text)
        text = re.sub(r'[\u200B-\u200D\uFEFF\u00AD\u2060]', '', text)
        text = cls.normalize_bengali_unicode(text)
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        text = text.replace('॥', '।')
        text = re.sub(r'[0-9\u09E6-\u09EF]', ' ', text)

        cleaned_chars = []
        for char in text:
            code = ord(char)
            if (0x0980 <= code <= 0x09FF) or char in {'।', '\n', ' ', '?', '!', ',', '-', ';', '—'}:
                cleaned_chars.append(char)
            else:
                cleaned_chars.append(' ')

        text = ''.join(cleaned_chars)
        text = re.sub(r'[ \t\f\v]+', ' ', text)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        return text.strip()

    @classmethod
    def tokenize_line_to_words(cls, line: str) -> List[str]:
        raw_words = line.split()
        cleaned_words: List[str] = []

        for w in raw_words:
            w_norm = cls.normalize_bengali_unicode(w)
            #  also split on , ; । so 'পাখি,আকাশ' becomes two tokens
            parts = re.split(r'[—–\-_/|,;।!?]+', w_norm)

            for part in parts:
                w_clean = re.sub(rf'^[^{cls.BENGALI_ALL}]+', '', part)
                w_clean = re.sub(rf'[^{cls.BENGALI_ALL}]+$', '', w_clean)
                if not w_clean:
                    continue

                if len(w_clean) > 30:
                    continue

                if len(w_clean) >= 6 and len(w_clean) % 2 == 0:
                    half = len(w_clean) // 2
                    if w_clean[:half] == w_clean[half:]:
                        continue

                if len(w_clean) >= 6:
                    repeated = False
                    for l in (2, 3):
                        if len(w_clean) % l == 0:
                            chunk = len(w_clean) // l
                            if w_clean[:chunk] * l == w_clean:
                                repeated = True
                                break
                    if repeated:
                        continue

                cleaned_words.append(w_clean)

        merged_words: List[str] = []
        i = 0
        while i < len(cleaned_words):
            curr = cleaned_words[i]
            if (i + 1 < len(cleaned_words)
                    and len(curr) == 1
                    and curr not in {'এ', 'ও', 'ঐ'}
                    and re.match(r'^[\u09DF\u09BE-\u09CD]', cleaned_words[i + 1])):
                merged = curr + cleaned_words[i + 1]
                if cls.is_valid_bengali_word(merged):
                    merged_words.append(merged)
                i += 2
            else:
                if cls.is_valid_bengali_word(curr):
                    merged_words.append(curr)
                i += 1

        return merged_words

    @staticmethod
    def _poem_hash(cleaned_text: str) -> str:
        """Fingerprint of a cleaned poem (whitespace/punctuation-insensitive)."""
        norm = re.sub(r'[\s।,;!?\-—]+', '', cleaned_text)
        return hashlib.md5(norm.encode('utf-8')).hexdigest()

    def process_corpus(self, remove_duplicates: bool = True):
        if self.raw_df is None:
            self.load_and_inspect_dataset()

        df = self.raw_df.copy()

        if remove_duplicates and 'poem' in df.columns:
            initial_count = len(df)
            df = df.drop_duplicates(subset=['poem']).reset_index(drop=True)
            print(f"Deduplicated dataset: {initial_count} -> {len(df)} records "
                  f"({initial_count - len(df)} duplicates removed)")

        self.cleaned_df = df
        self.corpus_lines = []
        self.corpus_poems = []
        self.corpus_labeled_poems = []
        self.line_end_counts = Counter()
        self.line_start_counts = Counter()
        self.vocab_counts = Counter()
        self._seen_poem_hashes = set()
        dup_skipped = 0

        for idx, row in df.iterrows():
            poem_text = str(row.get('poem', ''))
            cleaned_text = self.clean_bengali_text(poem_text)
            label = str(row.get('label', 'Miscellaneous')).strip() or 'Miscellaneous'

            if not cleaned_text:
                continue

            h = self._poem_hash(cleaned_text)
            if h in self._seen_poem_hashes:
                dup_skipped += 1
                continue
            self._seen_poem_hashes.add(h)

            raw_lines = re.split(r'[\n।!?]+', cleaned_text)
            current_poem_lines = []

            for line in raw_lines:
                words = self.tokenize_line_to_words(line)
                if len(words) >= 3:
                    self.corpus_lines.append(words)
                    current_poem_lines.append(words)
                    self.vocab_counts.update(words)
                    self.line_start_counts[words[0]] += 1
                    self.line_end_counts[words[-1]] += 1

            if current_poem_lines:
                self.corpus_poems.append(current_poem_lines)
                self.corpus_labeled_poems.append({
                    'label': label,
                    'lines': current_poem_lines
                })

        if dup_skipped:
            print(f"  Near-duplicate poems skipped: {dup_skipped:,}")
        print(f"Corpus Preprocessed:")
        print(f"  Total Poems / Documents : {len(self.corpus_poems):,}")
        print(f"  Total Verse Lines       : {len(self.corpus_lines):,}")
        print(f"  Total Tokens (words)    : {sum(self.vocab_counts.values()):,}")
        print(f"  Unique Vocabulary Size  : {len(self.vocab_counts):,}")
        print(f"  Unique Line Start Words : {len(self.line_start_counts):,}")
        print(f"  Unique Line End Words   : {len(self.line_end_counts):,}")

        return self.corpus_lines, self.corpus_poems, self.line_end_counts

    def load_extra_poems(
        self,
        extra_dir: str,
        default_label: str = "Miscellaneous",
        apply_label_map: bool = True,
    ) -> int:
        if not os.path.isdir(extra_dir):
            print(f"Note: '{extra_dir}' not found — skipping extra data.")
            return 0

        added = 0
        scanned_dirs = 0
        skipped = 0
        dup_skipped = 0
        unmapped_classes = Counter()

        for poet_dir in sorted(os.listdir(extra_dir)):
            poet_path = os.path.join(extra_dir, poet_dir)
            if not os.path.isdir(poet_path):
                continue

            for title_dir in sorted(os.listdir(poet_path)):
                poem_path = os.path.join(poet_path, title_dir)
                if not os.path.isdir(poem_path):
                    continue

                scanned_dirs += 1

                class_path = os.path.join(poem_path, "CLASS.txt")
                raw_class = ""
                if os.path.exists(class_path):
                    try:
                        with open(class_path, "r", encoding="utf-8") as f:
                            raw_class = f.read().strip()
                    except Exception:
                        pass

                if apply_label_map and raw_class:
                    key = raw_class.replace("/", " ").strip()
                    label = self.EXTRA_LABEL_MAP.get(key)
                    if label is None:
                        label = self.EXTRA_LABEL_MAP.get(key.title())
                    if label is None:
                        unmapped_classes[key] += 1
                        label = default_label
                else:
                    label = raw_class.replace("/", " ").strip() or default_label

                poem_text = None
                for fname in os.listdir(poem_path):
                    if not fname.lower().endswith(".txt"):
                        continue
                    if fname in ("CLASS.txt", "SOURCE.txt"):
                        continue
                    try:
                        with open(os.path.join(poem_path, fname),
                                  "r", encoding="utf-8", errors="replace") as f:
                            poem_text = f.read()
                        break
                    except Exception:
                        continue

                if not poem_text or not poem_text.strip():
                    skipped += 1
                    continue

                cleaned = self.clean_bengali_text(poem_text)
                if not cleaned:
                    skipped += 1
                    continue

                # FIX: skip poems already seen (ODS or another extra folder)
                h = self._poem_hash(cleaned)
                if h in self._seen_poem_hashes:
                    dup_skipped += 1
                    continue
                self._seen_poem_hashes.add(h)

                # FIX: identical rules to process_corpus (split on \n । ! ?, min 3 words)
                lines: List[List[str]] = []
                for raw_line in re.split(r'[\n।!?]+', cleaned):
                    words = self.tokenize_line_to_words(raw_line)
                    if len(words) >= 3:
                        lines.append(words)

                if not lines:
                    skipped += 1
                    continue

                self.corpus_poems.append(lines)
                self.corpus_labeled_poems.append({
                    "label": label,
                    "lines": lines,
                })
                for ln in lines:
                    self.corpus_lines.append(ln)
                    self.vocab_counts.update(ln)
                    self.line_start_counts[ln[0]] += 1
                    self.line_end_counts[ln[-1]] += 1
                added += 1

        print(f"Extra data scan complete:")
        print(f"  Poem dirs scanned : {scanned_dirs:,}")
        print(f"  Poems skipped     : {skipped:,}")
        print(f"  Duplicates skipped: {dup_skipped:,}")
        print(f"  Poems added       : {added:,}")
        print(f"  Corpus lines      : {len(self.corpus_lines):,} (cumulative)")
        print(f"  Vocab (unique)    : {len(self.vocab_counts):,} (cumulative)")

        all_labels = Counter(p["label"] for p in self.corpus_labeled_poems)
        print(f"\n  Label distribution (full corpus, {len(all_labels)} labels):")
        for lbl, cnt in all_labels.most_common():
            print(f"      {lbl:<20}: {cnt:,}")

        if unmapped_classes:
            print(f"\n  Unmapped raw classes → '{default_label}':")
            for cls, cnt in unmapped_classes.most_common(15):
                print(f"      {cls:<25}: {cnt:,}")

        return added

    def train_val_split(self, val_ratio: float = 0.05, seed: int = 42):
        """
        Split at POEM level (never line level) so no poem leaks into validation.
        Returns (train_poems, val_poems); each item is {'label':..., 'lines': [...]}.
        """
        poems = list(self.corpus_labeled_poems)
        rng = random.Random(seed)
        rng.shuffle(poems)
        n_val = max(1, int(len(poems) * val_ratio)) if poems else 0
        return poems[n_val:], poems[:n_val]

    def save_line_counts(self, path: str = "models/line_counts.pkl"):
        """Persist line start/end counts so the web app can restore them."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"line_start_counts": self.line_start_counts,
                         "line_end_counts": self.line_end_counts}, f)
        print(f"Saved line counts -> {path}")

    @staticmethod
    def load_line_counts(path: str = "models/line_counts.pkl"):
        """Returns (line_start_counts, line_end_counts); empty Counters if missing."""
        if not os.path.exists(path):
            return Counter(), Counter()
        with open(path, "rb") as f:
            d = pickle.load(f)
        return d.get("line_start_counts", Counter()), d.get("line_end_counts", Counter())

    def get_corpus_statistics(self) -> Dict[str, Any]:
        total_tokens = sum(self.vocab_counts.values())
        unique_tokens = len(self.vocab_counts)
        ttr = (unique_tokens / total_tokens) if total_tokens > 0 else 0.0
        top_20_words = self.vocab_counts.most_common(20)

        stopword_tokens_count = sum(cnt for w, cnt in self.vocab_counts.items()
                                    if w in BENGALI_STOP_WORDS)
        content_tokens_count = total_tokens - stopword_tokens_count

        return {
            "total_documents": len(self.corpus_poems),
            "total_lines": len(self.corpus_lines),
            "total_tokens": total_tokens,
            "vocabulary_size": unique_tokens,
            "type_token_ratio": round(ttr, 4),
            "stopword_tokens_ratio": round(stopword_tokens_count / total_tokens, 4) if total_tokens else 0,
            "content_tokens_ratio": round(content_tokens_count / total_tokens, 4) if total_tokens else 0,
            "top_20_frequent_words": top_20_words,
            "average_tokens_per_line": round(total_tokens / len(self.corpus_lines), 2) if self.corpus_lines else 0
        }

    @staticmethod
    def is_stopword(word: str) -> bool:
        return word in BENGALI_STOP_WORDS

    @staticmethod
    def filter_stopwords(tokens: List[str]) -> List[str]:
        return [w for w in tokens if w not in BENGALI_STOP_WORDS]