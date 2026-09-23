import os
import argparse

from src.preprocessing import BengaliPreprocessor
from src.vocabulary import Vocabulary
from src.lstm_model import BanglaLSTMTrainer
from src.lstm_generator import CorpusTopicStats


# ----------------------------------------------------------------------
# Paths and constants
# ----------------------------------------------------------------------
DATA_PATH = os.path.join("data", "SAHITTO.ods")
EXTRA_DIR = r"E:\4-1\Poems\Bengali-Poem-Dataset\dataset"
MODELS_DIR = "models"

# Separate output paths so the baseline's files are never overwritten.
VOCAB_PATH        = os.path.join(MODELS_DIR, "vocab_min5.pkl")
LINE_COUNTS_PATH  = os.path.join(MODELS_DIR, "line_counts_min5.pkl")
TOPIC_STATS_PATH  = os.path.join(MODELS_DIR, "topic_stats_min5.pkl")
DEFAULT_OUT       = os.path.join(MODELS_DIR, "lstm_model_topic.pt")

MIN_FREQ = 5        # vocabulary frequency cutoff


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
ap = argparse.ArgumentParser(
    description="Retrain the topic-conditioned LSTM with a clean vocabulary "
                "and stronger regularization. Baseline model untouched."
)
ap.add_argument("--counts-only", action="store_true",
                help="only rebuild vocab + counts, do not train")
ap.add_argument("--out", default=DEFAULT_OUT,
                help="where to save the new LSTM checkpoint")
ap.add_argument("--overwrite", action="store_true",
                help="allow replacing an existing --out file")
args = ap.parse_args()

if not args.counts_only and os.path.exists(args.out) and not args.overwrite:
    raise SystemExit(
        f"{args.out} already exists.  Pick another --out or pass --overwrite."
    )


# ----------------------------------------------------------------------
# Step 1: Corpus preprocessing (same rules as baseline)
# ----------------------------------------------------------------------
print("=" * 60)
print("  Step 1/4: Corpus preprocessing")
print("=" * 60)

pre = BengaliPreprocessor(DATA_PATH)
pre.process_corpus(remove_duplicates=True)
pre.load_extra_poems(extra_dir=EXTRA_DIR, apply_label_map=True)

print(f"\n  Total poems : {len(pre.corpus_labeled_poems):,}")
print(f"  Total lines : {len(pre.corpus_lines):,}")


# ----------------------------------------------------------------------
# Step 2: Fresh vocabulary with min_freq=5
# ----------------------------------------------------------------------
print("\n" + "=" * 60)
print(f"  Step 2/4: Building fresh vocabulary (min_freq={MIN_FREQ})")
print("=" * 60)

vocab = Vocabulary(min_freq=MIN_FREQ)
vocab.build_vocabulary(pre.corpus_lines)
vocab.save(VOCAB_PATH)

print(f"  Vocabulary  : {vocab.vocab_size:,} tokens")
print(f"  Saved to    : {VOCAB_PATH}")
print(f"  Note        : the baseline vocab.pkl is untouched")


# ----------------------------------------------------------------------
# Step 3: Line counts and topic stats (separate files)
# ----------------------------------------------------------------------
print("\n" + "=" * 60)
print("  Step 3/4: Saving line counts and topic stats")
print("=" * 60)

pre.save_line_counts(LINE_COUNTS_PATH)
print(f"  Line counts -> {LINE_COUNTS_PATH}")

CorpusTopicStats(
    [p["lines"] for p in pre.corpus_labeled_poems]
).save(TOPIC_STATS_PATH)
print(f"  Topic stats -> {TOPIC_STATS_PATH}")


# ----------------------------------------------------------------------
# Step 4: Train the topic-conditioned LSTM
# ----------------------------------------------------------------------
if args.counts_only:
    print("\n--counts-only specified.  Exiting before training.")
    raise SystemExit(0)

print("\n" + "=" * 60)
print("  Step 4/4: Training topic-conditioned LSTM")
print("=" * 60)

train_poems, val_poems = pre.train_val_split(val_ratio=0.05, seed=42)
print(f"  Train poems     : {len(train_poems):,}")
print(f"  Validation poems: {len(val_poems):,}")

trainer = BanglaLSTMTrainer(
    vocab=vocab,
    seq_len=80,
    embed_dim=128,
    hidden_dim=256,
    num_layers=2,
    dropout=0.5,        # was 0.4  -- stronger regularization
    batch_size=64,      # was 32   -- smoother gradients
    epochs=20,          # was 40   -- early stopping will cut this short
    lr=5e-4,            # was 1e-3 -- slower, better generalization
    patience=3,         # was 6    -- stop sooner when val stops improving
)

report = trainer.train_poems(train_poems, val_poems)
trainer.save(args.out)

print("\n" + "=" * 60)
print("  Done.")
print("=" * 60)
print(f"  New model        : {args.out}")
print(f"  New vocab        : {VOCAB_PATH}")
print(f"  New line counts  : {LINE_COUNTS_PATH}")
print(f"  New topic stats  : {TOPIC_STATS_PATH}")
print(f"  Baseline         : models/lstm_model.pt  (untouched)")
print(f"                     models/vocab.pkl      (untouched)")
print(f"                     models/line_counts.pkl (untouched)")
print(f"                     models/topic_stats.pkl (untouched)")

print("\nTo test the new (topic-conditioned) model in this terminal:")
print(f'  PowerShell :  $env:LSTM_MODEL         = "{args.out}"')
print(f'  PowerShell :  $env:VOCAB_PATH         = "{VOCAB_PATH}"')
print(f'  PowerShell :  $env:LINE_COUNTS_PATH   = "{LINE_COUNTS_PATH}"')
print(f'  PowerShell :  $env:TOPIC_STATS_PATH   = "{TOPIC_STATS_PATH}"')
print("  cmd        :  set LSTM_MODEL=" + args.out)
print("  cmd        :  set VOCAB_PATH=" + VOCAB_PATH)
print("  cmd        :  set LINE_COUNTS_PATH=" + LINE_COUNTS_PATH)
print("  cmd        :  set TOPIC_STATS_PATH=" + TOPIC_STATS_PATH)
print("  then run   :  python check_topic.py")
print("  To go back :  close the terminal, or unset the variables with")
print("                Remove-Item Env:\\LSTM_MODEL, Env:\\VOCAB_PATH, "
      "Env:\\LINE_COUNTS_PATH, Env:\\TOPIC_STATS_PATH")