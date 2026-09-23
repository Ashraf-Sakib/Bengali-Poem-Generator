from __future__ import annotations

import math
import os
import random
from collections import Counter
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_label_mapping(labeled_poems):
    labels = sorted({
        str(poem.get("label", "Miscellaneous")).strip() or "Miscellaneous"
        for poem in labeled_poems
    })
    label2idx = {label: idx for idx, label in enumerate(labels)}
    idx2label = {idx: label for label, idx in label2idx.items()}
    return label2idx, idx2label


class LabelConditionedPoetryDataset(Dataset):
    def __init__(
        self,
        labeled_poems: List[Dict[str, Any]],
        vocab,
        seq_len: int = 32,
        label2idx: Dict[str, int] | None = None,
        stride: int | None = None,
    ):
        self.vocab = vocab
        self.seq_len = seq_len
        self.stride = stride if stride is not None else 8

        if label2idx is None:
            self.label2idx, self.idx2label = build_label_mapping(labeled_poems)
        else:
            self.label2idx = dict(label2idx)
            self.idx2label = {idx: label for label, idx in self.label2idx.items()}

        self.inputs: List[List[int]] = []
        self.targets: List[List[int]] = []
        self.labels: List[int] = []

        self._build_sequences(labeled_poems)

    def _get_vocab_id(self, token: str) -> int:
        if hasattr(self.vocab, "word2idx"):
            if token in self.vocab.word2idx:
                return self.vocab.word2idx[token]
            if hasattr(self.vocab, "unk_idx"):
                return self.vocab.unk_idx
            if "<UNK>" in self.vocab.word2idx:
                return self.vocab.word2idx["<UNK>"]
        raise AttributeError("Could not convert word to vocab id.")

    def _special_id(self, token: str) -> int:
        if hasattr(self.vocab, "word2idx") and token in self.vocab.word2idx:
            return self.vocab.word2idx[token]
        attr_name = {
            "<PAD>": "pad_idx",
            "<UNK>": "unk_idx",
            "<START>": "start_idx",
            "<END>": "end_idx",
            "<LINE>": "line_idx",
        }.get(token)
        if attr_name and hasattr(self.vocab, attr_name):
            return getattr(self.vocab, attr_name)
        raise KeyError(f"Special token {token} not found in vocabulary.")

    def _build_sequences(self, labeled_poems: List[Dict[str, Any]]) -> None:
        start_id = self._special_id("<START>")
        end_id   = self._special_id("<END>")
        pad_id   = self._special_id("<PAD>")
        try:
            line_id = self._special_id("<LINE>")
        except KeyError:
            line_id = end_id

        for poem in labeled_poems:
            label = (str(poem.get("label", "Miscellaneous")).strip()
                     or "Miscellaneous")
            if label not in self.label2idx:
                continue
            label_id = self.label2idx[label]

            poem_lines = poem.get("lines", [])
            poem_ids: List[int] = []

            for line in poem_lines:
                if not line:
                    continue
                line_ids = [self._get_vocab_id(w) for w in line]
                if not line_ids:
                    continue
                poem_ids.append(start_id)
                poem_ids.extend(line_ids)
                poem_ids.append(end_id)
                poem_ids.append(line_id)

            while poem_ids and poem_ids[-1] == line_id:
                poem_ids.pop()

            if len(poem_ids) < 4:
                continue

            if len(poem_ids) <= self.seq_len:
                x = [pad_id] * (self.seq_len - len(poem_ids)) + poem_ids
                y = x[1:] + [pad_id]
                self.inputs.append(x)
                self.targets.append(y)
                self.labels.append(label_id)
                continue

            for i in range(0, len(poem_ids) - self.seq_len, self.stride):
                x = poem_ids[i:i + self.seq_len]
                y = poem_ids[i + 1:i + self.seq_len + 1]
                self.inputs.append(x)
                self.targets.append(y)
                self.labels.append(label_id)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.inputs[idx], dtype=torch.long),
            torch.tensor(self.targets[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


class LabelConditionedBanglaLSTMModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_labels: int,
        embed_dim: int = 128,
        label_embed_dim: int = 16,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.5,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_labels = num_labels
        self.embed_dim = embed_dim
        self.label_embed_dim = label_embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.label_embedding = nn.Embedding(num_labels, label_embed_dim)

        self.label_proj = nn.Sequential(
            nn.Linear(label_embed_dim, label_embed_dim),
            nn.Tanh(),
            nn.Dropout(dropout * 0.5),
        )

        self.lstm = nn.LSTM(
            input_size=embed_dim + label_embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.label_embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.embedding.weight[self.pad_idx].fill_(0)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x, labels, hidden=None):
        word_emb = self.embedding(x)
        label_emb = self.label_embedding(labels).unsqueeze(1)
        label_emb = self.label_proj(label_emb)
        label_emb = label_emb.expand(-1, x.size(1), -1)
        combined = torch.cat([word_emb, label_emb], dim=-1)
        output, hidden = self.lstm(combined, hidden)
        output = self.dropout(output)
        logits = self.fc(output)
        return logits, hidden


class LabelConditionedBanglaLSTMTrainer:
    def __init__(
        self,
        vocab,
        seq_len: int = 32,
        embed_dim: int = 128,
        label_embed_dim: int = 16,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.5,
        batch_size: int = 64,
        epochs: int = 30,
        learning_rate: float = 0.0005,
        weight_decay: float = 1e-4,
        patience: int = 6,
        validation_split: float = 0.10,
        seed: int = 42,
        stride: int | None = None,
    ):
        self.vocab = vocab
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.label_embed_dim = label_embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.validation_split = validation_split
        self.seed = seed
        self.stride = stride if stride is not None else 8

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")

    def _split_poems(self, labeled_poems):
        set_seed(self.seed)
        by_label = {}
        for poem in labeled_poems:
            label = (str(poem.get("label", "Miscellaneous")).strip()
                     or "Miscellaneous")
            by_label.setdefault(label, []).append(poem)

        train_poems, val_poems = [], []
        for poems in by_label.values():
            poems = list(poems)
            random.shuffle(poems)
            val_count = max(1, int(len(poems) * self.validation_split))
            val_poems.extend(poems[:val_count])
            train_chunk = poems[val_count:]
            if len(poems) < 40:
                train_chunk = train_chunk * 3
            train_poems.extend(train_chunk)

        random.shuffle(train_poems)
        random.shuffle(val_poems)
        return train_poems, val_poems

    def _get_vocab_size(self) -> int:
        if hasattr(self.vocab, "word2idx"):
            return len(self.vocab.word2idx)
        if hasattr(self.vocab, "stoi"):
            return len(self.vocab.stoi)
        if hasattr(self.vocab, "__len__"):
            return len(self.vocab)
        raise AttributeError("Could not determine vocabulary size.")

    def _get_pad_idx(self) -> int:
        if hasattr(self.vocab, "pad_idx"):
            return self.vocab.pad_idx
        if hasattr(self.vocab, "word2idx"):
            return self.vocab.word2idx.get("<PAD>", 0)
        return 0

    @torch.no_grad()
    def _evaluate(self, model, loader, criterion):
        model.eval()
        total_loss = 0.0
        total_batches = 0
        for x, y, labels in loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            logits, _ = model(x, labels)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
            )
            total_loss += loss.item()
            total_batches += 1
        if total_batches == 0:
            return float("inf")
        return total_loss / total_batches

    def train(self, labeled_poems, save_path="models/lstm_label_model.pt"):
        set_seed(self.seed)
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

        print("\nPreparing label-conditioned dataset...")
        label2idx, idx2label = build_label_mapping(labeled_poems)
        print(f"Number of labels: {len(label2idx)}")
        print("Labels:")
        for idx in sorted(idx2label):
            print(f"  {idx}: {idx2label[idx]}")

        label_counts = Counter(
            (str(poem.get("label", "Miscellaneous")).strip() or "Miscellaneous")
            for poem in labeled_poems
        )
        print("\nLabel distribution:")
        for label in sorted(label_counts):
            print(f"  {label}: {label_counts[label]:,}")

        train_poems, val_poems = self._split_poems(labeled_poems)
        print(f"\nTraining poems: {len(train_poems):,}")
        print(f"Validation poems: {len(val_poems):,}")

        train_dataset = LabelConditionedPoetryDataset(
            train_poems, self.vocab, seq_len=self.seq_len,
            label2idx=label2idx, stride=self.stride,
        )
        val_dataset = LabelConditionedPoetryDataset(
            val_poems, self.vocab, seq_len=self.seq_len,
            label2idx=label2idx, stride=self.stride,
        )

        print(f"\nTraining sequences: {len(train_dataset):,}")
        print(f"Validation sequences: {len(val_dataset):,}")

        if len(train_dataset) == 0:
            raise RuntimeError("Training dataset is empty.")
        if len(val_dataset) == 0:
            raise RuntimeError("Validation dataset is empty.")

        use_pin_memory = torch.cuda.is_available()
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                  shuffle=True, drop_last=True, pin_memory=use_pin_memory)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size,
                                shuffle=False, drop_last=False, pin_memory=use_pin_memory)

        vocab_size = self._get_vocab_size()
        pad_idx = self._get_pad_idx()

        model = LabelConditionedBanglaLSTMModel(
            vocab_size=vocab_size,
            num_labels=len(label2idx),
            embed_dim=self.embed_dim,
            label_embed_dim=self.label_embed_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            pad_idx=pad_idx,
        ).to(self.device)

        total_params = sum(p.numel() for p in model.parameters())
        print(f"\nTotal parameters: {total_params:,}")

        criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate,
                                     weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2,
        )

        best_val_loss = float("inf")
        epochs_without_improvement = 0

        print("\nStarting training...\n")

        for epoch in range(1, self.epochs + 1):
            model.train()
            total_train_loss = 0.0
            train_batches = 0

            for x, y, labels in train_loader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                logits, _ = model(x, labels)
                loss = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    y.reshape(-1),
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_train_loss += loss.item()
                train_batches += 1

            if train_batches == 0:
                raise RuntimeError("No training batches were produced.")

            train_loss = total_train_loss / train_batches
            val_loss = self._evaluate(model, val_loader, criterion)
            scheduler.step(val_loss)

            train_ppl = math.exp(min(train_loss, 20))
            val_ppl = math.exp(min(val_loss, 20))
            current_lr = optimizer.param_groups[0]["lr"]

            print(f"Epoch {epoch:02d}/{self.epochs} | "
                  f"Train Loss {train_loss:.4f} | Train PPL {train_ppl:.2f} | "
                  f"Val Loss {val_loss:.4f} | Val PPL {val_ppl:.2f} | "
                  f"LR {current_lr:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0

                checkpoint = {
                    "state_dict": model.state_dict(),
                    "label2idx": label2idx,
                    "idx2label": idx2label,
                    "lstm_config": {
                        "vocab_size": vocab_size,
                        "num_labels": len(label2idx),
                        "embed_dim": self.embed_dim,
                        "label_embed_dim": self.label_embed_dim,
                        "hidden_dim": self.hidden_dim,
                        "num_layers": self.num_layers,
                        "dropout": self.dropout,
                        "pad_idx": pad_idx,
                        "seq_len": self.seq_len,
                        "stride": self.stride,
                        "version": "v3",
                    },
                    "best_val_loss": best_val_loss,
                    "best_val_ppl": math.exp(min(best_val_loss, 20)),
                }
                torch.save(checkpoint, save_path)
                print(f"  Best model saved to {save_path}")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    print("\nEarly stopping.")
                    break

        print("\nTraining complete.")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Best validation PPL: {math.exp(min(best_val_loss, 20)):.2f}")
        print(f"Model saved at: {save_path}")

        return model, label2idx, idx2label

    @staticmethod
    def load_model(model_path: str, vocab, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        config = checkpoint["lstm_config"]

        model = LabelConditionedBanglaLSTMModel(
            vocab_size=config["vocab_size"],
            num_labels=config["num_labels"],
            embed_dim=config["embed_dim"],
            label_embed_dim=config["label_embed_dim"],
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
            pad_idx=config["pad_idx"],
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()
        return model, checkpoint["label2idx"], checkpoint["idx2label"]


def train_label_model(
    labeled_poems,
    vocab,
    save_path="models/lstm_label_model.pt",
    seq_len=32,
    epochs=40,
    batch_size=64,
    stride=8,
    weight_decay=1e-4,
):
    trainer = LabelConditionedBanglaLSTMTrainer(
        vocab=vocab,
        seq_len=seq_len,
        embed_dim=128,
        label_embed_dim=16,
        hidden_dim=256,
        num_layers=2,
        dropout=0.5,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=0.0005,
        weight_decay=weight_decay,
        patience=6,
        validation_split=0.10,
        seed=42,
        stride=stride,
    )
    return trainer.train(labeled_poems=labeled_poems, save_path=save_path)