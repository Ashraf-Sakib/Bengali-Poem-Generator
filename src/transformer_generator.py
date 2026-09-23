"""
src/transformer_generator.py
-----------------------------
Keyword-based Bangla Poem Generator using the label-FREE Transformer.

Theme/topic control works the same way as the plain LSTM generator:
  1. Keyword tokens are fed as the opening prompt (priming).
  2. Top-k / top-p sampling with a repetition penalty generates the rest.
"""

import math
import unicodedata
from collections import Counter
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from src.transformer_model import BanglaPoemTransformer
from src.vocabulary import Vocabulary


class TransformerPoemGenerator:

    def __init__(
        self,
        model: BanglaPoemTransformer,
        vocab: Vocabulary,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.vocab = vocab
        self.device = device or next(model.parameters()).device
        self.model.eval()

    # ── Sampling helper ────────────────────────────────────────────────────────

    @torch.no_grad()
    def _sample_next(
        self,
        logits: torch.Tensor,
        generated_ids: List[int],
        current_line_ids: List[int],
        min_words_per_line: int,
        temperature: float = 0.9,
        top_k: int = 30,
        top_p: float = 0.92,
        rep_penalty: float = 1.15,
        rep_window: int = 40,
    ) -> int:
        logits = logits.clone()

        pad_id   = self.vocab.pad_idx
        unk_id   = self.vocab.unk_idx
        start_id = self.vocab.start_idx
        end_id   = self.vocab.end_idx
        line_id  = self.vocab.line_idx

        # Always suppress structural tokens
        for tid in (pad_id, unk_id, start_id, line_id):
            logits[tid] = -1e9

        # Suppress <END> until the line is long enough
        if len(current_line_ids) < min_words_per_line:
            logits[end_id] = -1e9

        # Repetition penalty over recent tokens
        if generated_ids:
            recent = generated_ids[-rep_window:]
            counts = Counter(recent)
            for tid, c in counts.items():
                if tid in (pad_id, unk_id, start_id, end_id, line_id):
                    continue
                if logits[tid] > 0:
                    logits[tid] /= (rep_penalty ** c)
                else:
                    logits[tid] *= (rep_penalty ** c)

        logits = logits / max(temperature, 0.05)

        # Top-k filtering
        if top_k > 0:
            kth = torch.topk(logits, min(top_k, logits.size(-1))).values[-1]
            logits[logits < kth] = -1e9

        probs = F.softmax(logits, dim=-1)

        # Nucleus (top-p) filtering
        if top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cum = torch.cumsum(sorted_probs, dim=-1)
            mask = (cum - sorted_probs) > top_p
            sorted_probs[mask] = 0.0
            probs = torch.zeros_like(probs).scatter_(-1, sorted_idx, sorted_probs)
            s = probs.sum()
            if s > 0:
                probs = probs / s

        if probs.sum() <= 0 or torch.isnan(probs).any():
            return end_id

        return int(torch.multinomial(probs, num_samples=1).item())

    # ── Main generation method ─────────────────────────────────────────────────

    @torch.no_grad()
    def generate_poem(
        self,
        keyword: str = "",
        num_lines: int = 4,
        min_words_per_line: int = 4,
        max_words_per_line: int = 8,
        temperature: float = 0.9,
        top_k: int = 30,
        top_p: float = 0.92,
        rep_penalty: float = 1.15,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate a Bengali poem primed with *keyword*.

        Args:
            keyword            : theme word(s) to prime generation (space-separated)
            num_lines          : number of lines to generate (2–6)
            min_words_per_line : minimum content words per line
            max_words_per_line : maximum content words per line
            temperature        : softmax temperature (lower = more focused)
            top_k              : keep only the top-k logits before sampling
            top_p              : nucleus probability mass cutoff
            rep_penalty        : repetition penalty multiplier (>1 discourages repeats)
            seed               : optional RNG seed for reproducibility
        """
        if seed is not None:
            torch.manual_seed(seed)

        num_lines = max(2, min(num_lines, 6))

        start_id = self.vocab.start_idx
        end_id   = self.vocab.end_idx
        line_id  = self.vocab.line_idx
        max_len  = self.model.max_len

        # ── Build keyword prompt tokens ────────────────────────────────────────
        prompt_ids: List[int] = []
        if keyword and keyword.strip():
            for raw_w in keyword.strip().split():
                w = unicodedata.normalize("NFC", raw_w.strip())
                if not w:
                    continue
                wid = self.vocab.word2idx.get(w, self.vocab.unk_idx)
                if wid != self.vocab.unk_idx:
                    prompt_ids.append(wid)

        # ── Token buffer: start with <START> [+ prompt] ───────────────────────
        tokens: List[int] = [start_id] + prompt_ids
        lines: List[List[int]] = []
        current_line_ids: List[int] = list(prompt_ids)   # first line starts with keyword
        line_count = 0

        for _ in range(max_len - 1):
            if line_count >= num_lines:
                break

            inp = torch.tensor([tokens], dtype=torch.long, device=self.device)
            attn_mask = torch.ones_like(inp)
            logits, _ = self.model(inp, attn_mask)
            last_logits = logits[0, -1, :]

            next_id = self._sample_next(
                logits=last_logits,
                generated_ids=tokens,
                current_line_ids=current_line_ids,
                min_words_per_line=min_words_per_line,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                rep_penalty=rep_penalty,
            )

            # ── Line boundary ──────────────────────────────────────────────────
            if next_id == end_id or len(current_line_ids) >= max_words_per_line:
                if current_line_ids:
                    lines.append(list(current_line_ids))
                tokens.append(end_id)
                tokens.append(line_id)
                current_line_ids = []
                line_count += 1
                if line_count >= num_lines:
                    break
                tokens.append(start_id)
                continue

            tokens.append(next_id)
            current_line_ids.append(next_id)

        formatted_lines: List[str] = []
        for i, line_ids in enumerate(lines):
            words = [
                self.vocab.idx_to_word(t)
                for t in line_ids
                if self.vocab.idx_to_word(t) not in self.vocab.SPECIAL_TOKENS
            ]
            line_str = " ".join(words)
            if not line_str.strip():
                continue
            punctuation = "।" if i == len(lines) - 1 else ","
            formatted_lines.append(f"{line_str}{punctuation}")

        poem_text = "\n".join(formatted_lines)

        all_words = [
            w for ln in lines
            for t in ln
            for w in [self.vocab.idx_to_word(t)]
            if w not in self.vocab.SPECIAL_TOKENS
        ]
        total  = len(all_words)
        unique = len(set(all_words))
        ttr    = round(unique / total, 4) if total else 0.0

        return {
            "keyword":    keyword,
            "lines":      formatted_lines,
            "poem_text":  poem_text,
            "metrics": {
                "total_words":      total,
                "unique_words":     unique,
                "ttr":              ttr,
                "avg_words_per_line": round(total / max(1, len(formatted_lines)), 2),
            },
        }