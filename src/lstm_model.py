import os
import math
import time
import pickle
import random
from collections import Counter
from typing import List, Tuple, Optional, Dict, Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.vocabulary import Vocabulary
from src.preprocessing import BENGALI_STOP_WORDS


# Dataset 1 — line-level (used by main.py --retrain, the baseline path)

class PoetrySequenceDataset:

    GROUP_SIZE = 4

    def __init__(self, corpus_lines: List[List[str]], vocab: Vocabulary, seq_len: int = 32):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required.")

        self.vocab = vocab
        self.seq_len = seq_len
        self.inputs: List[List[int]] = []
        self.targets: List[List[int]] = []

        start_id = vocab.word2idx[vocab.START_TOKEN]
        end_id   = vocab.word2idx[vocab.END_TOKEN]
        pad_id   = vocab.word2idx[vocab.PAD_TOKEN]
        line_id  = vocab.word2idx.get(vocab.NEWLINE_TOKEN, end_id)

        for block_start in range(0, len(corpus_lines), self.GROUP_SIZE):
            block = corpus_lines[block_start:block_start + self.GROUP_SIZE]
            poem_ids: List[int] = []

            for line in block:
                if not line:
                    continue
                line_ids = vocab.encode(line)
                if not line_ids:
                    continue
                poem_ids.append(start_id)
                poem_ids.extend(line_ids)
                poem_ids.append(end_id)
                poem_ids.append(line_id)

            while poem_ids and poem_ids[-1] == line_id:
                poem_ids.pop()

            if len(poem_ids) < 3:
                continue

            if len(poem_ids) <= seq_len:
                x = [pad_id] * (seq_len - len(poem_ids)) + poem_ids
                y = x[1:] + [pad_id]
                self.inputs.append(x)
                self.targets.append(y)
                continue

            for i in range(0, len(poem_ids) - seq_len, 5):
                x = poem_ids[i:i + seq_len]
                y = poem_ids[i + 1:i + seq_len + 1]
                self.inputs.append(x)
                self.targets.append(y)

    def __len__(self) -> int:
        return len(self.inputs)

    def as_tensors(self):
        import torch
        X = torch.tensor(self.inputs,  dtype=torch.long)
        Y = torch.tensor(self.targets, dtype=torch.long)
        return X, Y


# Topic-word selection


def pick_topic_words(poems,
                     vocab,
                     max_words: int = 3,
                     min_df: int = 10,
                     max_df_ratio: float = 0.4):
  
    n = len(poems)
    df = Counter()
    for p in poems:
        df.update({w for line in p["lines"] for w in line})

    max_df = max(1, int(n * max_df_ratio))

    out = []
    for p in poems:
        tf = Counter(w for line in p["lines"] for w in line)
        scored = []
        for w, c in tf.items():
            if len(w) < 3:
                continue
            if w in BENGALI_STOP_WORDS:
                continue
            if not vocab.contains(w):
                continue
            if df[w] < min_df:            # too rare  -> reject
                continue
            if df[w] > max_df:            # too generic -> reject
                continue
            scored.append((c, w))
        scored.sort(reverse=True)
        out.append([w for _, w in scored[:max_words]])
    return out


class PoemTopicDataset:

    MAX_LINES = 6

    def __init__(self, poems, vocab, seq_len: int = 80, topic_words=None,
                 max_topic_words: int = 3, topic_dropout: float = 0.1,
                 repeats: int = 1, seed: int = 0):
        rng = random.Random(seed)
        start = vocab.word2idx[vocab.START_TOKEN]
        end   = vocab.word2idx[vocab.END_TOKEN]
        pad   = vocab.word2idx[vocab.PAD_TOKEN]
        line  = vocab.word2idx.get(vocab.NEWLINE_TOKEN, end)
        topic_words = topic_words or [[] for _ in poems]

        self.inputs: List[List[int]] = []
        self.targets: List[List[int]] = []

        for poem, topics in zip(poems, topic_words):
            lines = [l for l in poem["lines"] if l]
            for c in range(0, len(lines), self.MAX_LINES):
                chunk = lines[c:c + self.MAX_LINES]
                if len(chunk) < 2 and c > 0:
                    continue

                body: List[int] = []
                for ln in chunk:
                    body += [start] + vocab.encode(ln) + [end, line]
                while body and body[-1] == line:
                    body.pop()

                for _ in range(repeats):
                    prefix: List[int] = []
                    if topics and rng.random() >= topic_dropout:
                        k = rng.randint(1, min(max_topic_words, len(topics)))
                        prefix = [vocab.word_to_idx(w) for w in topics[:k]] + [line]

                    ids = (prefix + body)[:seq_len + 1]
                    if len(ids) < 4:
                        continue

                    x, y = ids[:-1], ids[1:]

                    for j in range(len(prefix)):
                        y[j] = pad

                    x += [pad] * (seq_len - len(x))
                    y += [pad] * (seq_len - len(y))
                    self.inputs.append(x)
                    self.targets.append(y)

    def __len__(self) -> int:
        return len(self.inputs)

    def as_tensors(self):
        X = torch.tensor(self.inputs, dtype=torch.long)
        Y = torch.tensor(self.targets, dtype=torch.long)
        return X, Y




class BanglaLSTMModel(nn.Module):
    """Two-layer word-level LSTM Language Model."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim:  int   = 128,
        hidden_dim: int   = 256,
        num_layers: int   = 2,
        dropout:    float = 0.3,
        pad_idx:    int   = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim  = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x, hidden=None):
        emb = self.dropout(self.embedding(x))
        out, hidden = self.lstm(emb, hidden)
        out = self.dropout(out)
        logits = self.fc(out)
        return logits, hidden

    def init_hidden(self, batch_size: int, device):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        return (h0, c0)



# Trainer


class BanglaLSTMTrainer:

    MODEL_CONFIG_KEY = "lstm_config"

    def __init__(
        self,
        vocab: Vocabulary,
        seq_len:    int   = 32,
        embed_dim:  int   = 128,
        hidden_dim: int   = 256,
        num_layers: int   = 2,
        dropout:    float = 0.4,
        batch_size: int   = 64,
        epochs:     int   = 25,
        lr:         float = 0.001,
        patience:   int   = 4,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for LSTM training.\n"
            )

        self.vocab      = vocab
        self.seq_len    = seq_len
        self.embed_dim  = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout    = dropout
        self.batch_size = batch_size
        self.epochs     = epochs
        self.lr         = lr
        self.patience   = patience

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[BanglaLSTMModel] = None
        self.topic_conditioned = False

 
    def _prepare_data(self, corpus_lines):
        dataset = PoetrySequenceDataset(corpus_lines, self.vocab, self.seq_len)
        X, Y = dataset.as_tensors()

        n = len(X)
        split = int(n * 0.9)
        indices = list(range(n))
        random.shuffle(indices)

        train_idx = indices[:split]
        val_idx   = indices[split:]

        X_train, Y_train = X[train_idx], Y[train_idx]
        X_val,   Y_val   = X[val_idx],   Y[val_idx]

        train_ds = torch.utils.data.TensorDataset(X_train, Y_train)
        val_ds   = torch.utils.data.TensorDataset(X_val,   Y_val)

        train_loader = DataLoader(train_ds, batch_size=self.batch_size,
                                  shuffle=True,  drop_last=True)
        val_loader   = DataLoader(val_ds,   batch_size=self.batch_size,
                                  shuffle=False, drop_last=False)
        return train_loader, val_loader

    #  topic-conditioned training 
    def train_poems(self, train_poems, val_poems, seq_len: int = 80):
     
        self.seq_len = seq_len
        self.topic_conditioned = True

        train_topics = pick_topic_words(train_poems, self.vocab,
                                        min_df=10, max_df_ratio=0.4)
        val_topics   = pick_topic_words(val_poems,   self.vocab,
                                        min_df=1,  max_df_ratio=1.0)

        tr = PoemTopicDataset(train_poems, self.vocab, seq_len, train_topics,
                              topic_dropout=0.1, repeats=5, seed=1)
        va = PoemTopicDataset(val_poems,   self.vocab, seq_len, val_topics,
                              topic_dropout=0.0, repeats=1, seed=2)

        Xt, Yt = tr.as_tensors()
        Xv, Yv = va.as_tensors()

        train_loader = DataLoader(
            torch.utils.data.TensorDataset(Xt, Yt),
            batch_size=self.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(
            torch.utils.data.TensorDataset(Xv, Yv),
            batch_size=self.batch_size, shuffle=False)

        return self.train(None, loaders=(train_loader, val_loader))

    
    @torch.no_grad()
    def evaluate_ppl(self, loader) -> Dict[str, Any]:
       
        self.model.eval()
        v = self.vocab
        pad = v.word2idx[v.PAD_TOKEN]
        start_id = v.word2idx[v.START_TOKEN]
        end_id   = v.word2idx[v.END_TOKEN]
        line_id  = v.word2idx.get(v.NEWLINE_TOKEN, end_id)

        kinds = {"START": start_id, "END": end_id}
        skip_ids = [start_id]
        if line_id != end_id:
            kinds["LINE"] = line_id
            skip_ids.append(line_id)

        special = torch.tensor(sorted(set(kinds.values())), device=self.device)
        skip    = torch.tensor(skip_ids, device=self.device)

        sums = {k: [0.0, 0] for k in [*kinds, "WORDS"]}
        tot  = {"all": [0.0, 0], "words_end": [0.0, 0]}

        for xb, yb in loader:
            xb = xb.to(self.device)
            yb = yb.to(self.device).reshape(-1)
            logits, _ = self.model(xb)
            ce = nn.functional.cross_entropy(
                logits.reshape(-1, v.vocab_size), yb, reduction="none")
            valid = yb != pad

            for name, idx in kinds.items():
                m = valid & (yb == idx)
                sums[name][0] += ce[m].sum().item()
                sums[name][1] += int(m.sum().item())

            m = valid & ~torch.isin(yb, special)
            sums["WORDS"][0] += ce[m].sum().item()
            sums["WORDS"][1] += int(m.sum().item())

            tot["all"][0] += ce[valid].sum().item()
            tot["all"][1] += int(valid.sum().item())

            m = valid & ~torch.isin(yb, skip)
            tot["words_end"][0] += ce[m].sum().item()
            tot["words_end"][1] += int(m.sum().item())

        def ppl(pair):
            return math.exp(min(pair[0] / max(pair[1], 1), 20))

        n_all = max(tot["all"][1], 1)
        report = {
            "ppl_all": round(ppl(tot["all"]), 1),
            "ppl_words_end": round(ppl(tot["words_end"]), 1),
            "by_type": {
                k: {"share_pct": round(100 * n / n_all, 1),
                    "mean_loss": round(s_ / max(n, 1), 3)}
                for k, (s_, n) in sums.items()
            },
        }
        print("  Validation perplexity")
        print(f"    all tokens    : {report['ppl_all']}")
        print(f"    words + <END> : {report['ppl_words_end']}")
        for k, d in report["by_type"].items():
            print(f"    {k:<6} {d['share_pct']:>5}% of targets, "
                  f"mean loss {d['mean_loss']}")
        if line_id == end_id:
            print("    (vocabulary has no <LINE> token: <END> doubles as the delimiter)")
        return report

    #  training loop 
    def train(self, corpus_lines: List[List[str]], loaders=None) -> Dict[str, Any]:
        print(f"\n{'='*60}")
        print(f"  Training LSTM Language Model"
              f"{' (topic-conditioned)' if self.topic_conditioned else ''}")
        print(f"  Device   : {self.device}")
        print(f"  Vocab    : {self.vocab.vocab_size:,} tokens")
        print(f"  Seq Len  : {self.seq_len}")
        print(f"  Embed    : {self.embed_dim}  Hidden: {self.hidden_dim}  "
              f"Layers: {self.num_layers}")
        print(f"  Epochs   : {self.epochs}  Batch: {self.batch_size}  LR: {self.lr}")
        print(f"{'='*60}\n")

        t0 = time.time()
        train_loader, val_loader = loaders if loaders else self._prepare_data(corpus_lines)
        print(f"  Dataset  : {len(train_loader.dataset):,} train / "
              f"{len(val_loader.dataset):,} val sequences\n")

        self.model = BanglaLSTMModel(
            vocab_size=self.vocab.vocab_size,
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            pad_idx=self.vocab.word2idx[self.vocab.PAD_TOKEN],
        ).to(self.device)

       
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=1e-5,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2
        )
        # label smoothing
        criterion = nn.CrossEntropyLoss(
            ignore_index=self.vocab.word2idx[self.vocab.PAD_TOKEN],
            label_smoothing=0.1,
        )

        best_val_loss = float('inf')
        best_epoch    = 0
        patience_ctr  = 0
        train_losses  = []
        val_losses    = []
        best_state    = None

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            n_batches  = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                logits, _ = self.model(xb)
                loss = criterion(
                    logits.reshape(-1, self.vocab.vocab_size),
                    yb.reshape(-1),
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches  += 1

            avg_train = epoch_loss / max(n_batches, 1)
            train_ppl = math.exp(min(avg_train, 20))

            self.model.eval()
            val_loss    = 0.0
            val_batches = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    logits, _ = self.model(xb)
                    loss = criterion(
                        logits.reshape(-1, self.vocab.vocab_size),
                        yb.reshape(-1),
                    )
                    val_loss    += loss.item()
                    val_batches += 1

            avg_val = val_loss / max(val_batches, 1)
            val_ppl = math.exp(min(avg_val, 20))
            scheduler.step(avg_val)

            train_losses.append(avg_train)
            val_losses.append(avg_val)

            elapsed = time.time() - t0
            print(
                f"  Epoch {epoch:>2}/{self.epochs}  "
                f"Train Loss: {avg_train:.4f} (PPL {train_ppl:.1f})  "
                f"Val Loss: {avg_val:.4f} (PPL {val_ppl:.1f})  "
                f"[{elapsed:.0f}s]"
            )

            if avg_val < best_val_loss - 1e-4:
                best_val_loss = avg_val
                best_epoch    = epoch
                patience_ctr  = 0
                best_state    = {k: v.cpu().clone()
                                 for k, v in self.model.state_dict().items()}
            else:
                patience_ctr += 1
                if patience_ctr >= self.patience:
                    print(f"\n  Early stopping at epoch {epoch}.")
                    break

        if best_state:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

        total_time = time.time() - t0
        best_ppl = math.exp(min(best_val_loss, 20))
        print(f"\n  Best Val Loss : {best_val_loss:.4f} (PPL {best_ppl:.1f})  "
              f"at epoch {best_epoch}")
        print(f"  Total Training Time: {total_time:.1f}s\n")

        val_report = self.evaluate_ppl(val_loader) if len(val_loader.dataset) > 0 else None

        return {
            "val_report": val_report,
            "train_losses": train_losses,
            "val_losses":   val_losses,
            "best_val_loss": best_val_loss,
            "best_epoch":    best_epoch,
            "best_perplexity": round(best_ppl, 2),
            "training_time_s": round(total_time, 1),
        }

    # persistence
    def save(self, model_path: str) -> None:
        if self.model is None:
            raise RuntimeError("Model has not been trained yet. Call train() first.")
        os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
        torch.save({
            "state_dict": self.model.state_dict(),
            self.MODEL_CONFIG_KEY: {
                "vocab_size":  self.vocab.vocab_size,
                "embed_dim":   self.embed_dim,
                "hidden_dim":  self.hidden_dim,
                "num_layers":  self.num_layers,
                "dropout":     self.dropout,
                "pad_idx":     self.vocab.word2idx[self.vocab.PAD_TOKEN],
                "seq_len":     self.seq_len,
                "topic_conditioned": self.topic_conditioned,
            }
        }, model_path)
        print(f"  LSTM model saved to: {model_path}")

    @classmethod
    def load_model(cls, model_path: str, vocab: Vocabulary) -> "BanglaLSTMModel":
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required. Install it with: pip install torch")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"LSTM model file not found: {model_path}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        cfg = checkpoint[cls.MODEL_CONFIG_KEY]

        if cfg["vocab_size"] != vocab.vocab_size:
            raise ValueError(
                f"Vocabulary size mismatch: checkpoint expects "
                f"{cfg['vocab_size']:,} tokens but current vocab has "
                f"{vocab.vocab_size:,}. Use the same vocab that was used at "
                f"training time (e.g. models/vocab_min5.pkl)."
            )

        model = BanglaLSTMModel(
            vocab_size=cfg["vocab_size"],
            embed_dim=cfg["embed_dim"],
            hidden_dim=cfg["hidden_dim"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
            pad_idx=cfg["pad_idx"],
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()
        model.topic_conditioned = bool(cfg.get("topic_conditioned", False))
        return model