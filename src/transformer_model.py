"""
src/transformer_model.py
------------------------
Label-FREE Bangla Poem Transformer (Causal LM).

Label conditioning removed — the model is now a pure next-token language model.
Theme/keyword control is done at generation time by priming the token sequence,
exactly the same way the plain LSTM generator works.
"""

import math
import os
import random
import time
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.transformer_dataset import PoemTransformerDataset, collate_poems


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Positional Encoding ────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ── Causal Self-Attention ──────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None):
        B, T, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)

        causal = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal, float("-inf"))

        if attn_mask is not None:
            pad_mask = attn_mask[:, None, None, :] == 0
            scores = scores.masked_fill(pad_mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.proj(out)


# ── Feed-Forward ───────────────────────────────────────────────────────────────

class FeedForward(nn.Module):

    def __init__(self, d_model: int, hidden_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_mult * d_model),
            nn.GELU(),
            nn.Linear(hidden_mult * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# ── Transformer Block ──────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, 4, dropout)

    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.norm1(x), attn_mask)
        x = x + self.ffn(self.norm2(x))
        return x


# ── Main Model (Label-FREE) ────────────────────────────────────────────────────

class BanglaPoemTransformer(nn.Module):
    """
    Causal (decoder-only) Transformer for Bangla poetry generation.
    No label conditioning — theme is controlled by keyword priming at inference.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        max_len: int = 128,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        self.pad_idx = pad_idx

        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: share token embedding and lm_head weights
        self.lm_head.weight = self.tok_emb.weight

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.tok_emb.weight[self.pad_idx].fill_(0)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.zeros_(self.norm.bias)
        nn.init.ones_(self.norm.weight)

    def forward(self, input_ids, attn_mask=None, hidden=None):
        """
        Args:
            input_ids : (B, T) token ids
            attn_mask : (B, T) 1=real token, 0=pad  [optional]
            hidden    : unused — kept for API compatibility with LSTM callers
        Returns:
            logits : (B, T, vocab_size)
            None   : (hidden state placeholder)
        """
        x = self.tok_emb(input_ids)
        x = self.pos_enc(x)

        for blk in self.blocks:
            x = blk(x, attn_mask)

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits, None


# ── Trainer ────────────────────────────────────────────────────────────────────

class BanglaTransformerTrainer:

    MODEL_CONFIG_KEY = "transformer_config"

    def __init__(
        self,
        vocab,
        max_len: int = 128,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1,
        batch_size: int = 32,
        epochs: int = 40,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        patience: int = 8,
        grad_clip: float = 1.0,
        label_smoothing: float = 0.05,
        validation_split: float = 0.08,
        seed: int = 42,
    ):
        self.vocab = vocab
        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.patience = patience
        self.grad_clip = grad_clip
        self.label_smoothing = label_smoothing
        self.validation_split = validation_split
        self.seed = seed

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        self.model = None

    def _split(self, poems):
        """Random train/val split at poem level."""
        set_seed(self.seed)
        poems = list(poems)
        random.shuffle(poems)
        n_val = max(1, int(len(poems) * self.validation_split))
        return poems[n_val:], poems[:n_val]

    def train(self, poems, save_path="models/transformer_model.pt"):
        """
        Args:
            poems : list of dicts with key "lines" (list of token lists)
                    OR list of list-of-token-lists.
                    Label fields are ignored if present.
        """
        set_seed(self.seed)
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

        # Normalise input: accept both labeled-poem dicts and raw line lists
        normalised = []
        for p in poems:
            if isinstance(p, dict):
                normalised.append(p)
            else:
                normalised.append({"lines": p})

        train_poems, val_poems = self._split(normalised)
        print(f"\nTrain poems: {len(train_poems):,}")
        print(f"Val poems  : {len(val_poems):,}")

        train_ds = PoemTransformerDataset(train_poems, self.vocab, max_len=self.max_len)
        val_ds   = PoemTransformerDataset(val_poems,   self.vocab, max_len=self.max_len)
        print(f"Train samples: {len(train_ds):,}")
        print(f"Val samples  : {len(val_ds):,}")

        if len(train_ds) == 0:
            raise RuntimeError("Training dataset is empty.")

        pad_id = self.vocab.pad_idx
        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True,
            drop_last=True, collate_fn=lambda b: collate_poems(b, pad_id),
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.batch_size, shuffle=False,
            drop_last=False, collate_fn=lambda b: collate_poems(b, pad_id),
        )

        self.model = BanglaPoemTransformer(
            vocab_size=self.vocab.vocab_size,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            max_len=self.max_len,
            dropout=self.dropout,
            pad_idx=pad_id,
        ).to(self.device)

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"\nTotal parameters: {n_params:,}")
        print(f"  d_model  : {self.d_model}")
        print(f"  n_heads  : {self.n_heads}")
        print(f"  n_layers : {self.n_layers}")
        print(f"  max_len  : {self.max_len}")
        print(f"  batch    : {self.batch_size}")
        print(f"  lr       : {self.lr}")

        criterion = nn.CrossEntropyLoss(
            ignore_index=pad_id,
            label_smoothing=self.label_smoothing,
        )
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
        )
        total_steps = len(train_loader) * self.epochs

        def lr_lambda(step):
            if step < self.warmup_steps:
                return step / max(1, self.warmup_steps)
            progress = (step - self.warmup_steps) / max(1, total_steps - self.warmup_steps)
            return max(0.05, 0.5 * (1 + math.cos(math.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        best_val = float("inf")
        patience_ctr = 0
        global_step = 0
        t0 = time.time()

        print("\nStarting training...\n")

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            tr_loss, tr_batches = 0.0, 0

            for batch in train_loader:
                input_ids = batch["input_ids"].to(self.device)
                labels    = batch["labels"].to(self.device)
                attn_mask = batch["attn_mask"].to(self.device)

                optimizer.zero_grad(set_to_none=True)
                logits, _ = self.model(input_ids, attn_mask)

                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()

                loss = criterion(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                optimizer.step()
                scheduler.step()
                global_step += 1

                tr_loss += loss.item()
                tr_batches += 1

            train_loss = tr_loss / max(1, tr_batches)

            self.model.eval()
            vl_loss, vl_batches = 0.0, 0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(self.device)
                    labels    = batch["labels"].to(self.device)
                    attn_mask = batch["attn_mask"].to(self.device)

                    logits, _ = self.model(input_ids, attn_mask)
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = labels[:, 1:].contiguous()
                    loss = criterion(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                    )
                    vl_loss += loss.item()
                    vl_batches += 1

            val_loss = vl_loss / max(1, vl_batches)
            tr_ppl = math.exp(min(train_loss, 20))
            vl_ppl = math.exp(min(val_loss, 20))
            lr_now = optimizer.param_groups[0]["lr"]

            print(f"Epoch {epoch:02d}/{self.epochs} | "
                  f"Train {train_loss:.4f} (PPL {tr_ppl:.1f}) | "
                  f"Val {val_loss:.4f} (PPL {vl_ppl:.1f}) | "
                  f"LR {lr_now:.2e} | "
                  f"{time.time()-t0:.0f}s")

            if val_loss < best_val:
                best_val = val_loss
                patience_ctr = 0

                torch.save({
                    "state_dict": self.model.state_dict(),
                    self.MODEL_CONFIG_KEY: {
                        "vocab_size": self.vocab.vocab_size,
                        "d_model":    self.d_model,
                        "n_heads":    self.n_heads,
                        "n_layers":   self.n_layers,
                        "max_len":    self.max_len,
                        "dropout":    self.dropout,
                        "pad_idx":    pad_id,
                    },
                    "best_val_loss": best_val,
                }, save_path)
                print(f"  Best model saved → {save_path}")
            else:
                patience_ctr += 1
                if patience_ctr >= self.patience:
                    print(f"\nEarly stopping (no improvement for {self.patience} epochs).")
                    break

        print(f"\nTraining complete. Best val loss: {best_val:.4f} "
              f"(PPL {math.exp(min(best_val, 20)):.2f})")
        return self.model

    @staticmethod
    def load_model(model_path: str, vocab, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        cfg = ckpt[BanglaTransformerTrainer.MODEL_CONFIG_KEY]

        model = BanglaPoemTransformer(
            vocab_size=cfg["vocab_size"],
            d_model=cfg["d_model"],
            n_heads=cfg["n_heads"],
            n_layers=cfg["n_layers"],
            max_len=cfg["max_len"],
            dropout=cfg["dropout"],
            pad_idx=cfg["pad_idx"],
        )
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        model.eval()
        return model