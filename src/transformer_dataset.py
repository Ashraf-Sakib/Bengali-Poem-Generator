"""
src/transformer_dataset.py
--------------------------
Dataset for the label-FREE Bangla Poem Transformer.
Label conditioning removed — poems are tokenised as plain sequences.
"""

from typing import Any, Dict, List, Tuple
import torch
from torch.utils.data import Dataset


class PoemTransformerDataset(Dataset):
    """
    Each poem becomes one sample: a flat token sequence of the form
        <START> w1 w2 ... <END> <LINE> <START> w1 w2 ... <END>
    truncated to max_len tokens.
    """

    def __init__(
        self,
        poems: List[Dict[str, Any]],
        vocab,
        max_len: int = 128,
    ):
        self.vocab = vocab
        self.max_len = max_len
        self.samples: List[List[int]] = []

        start_id = vocab.word2idx[vocab.START_TOKEN]
        end_id   = vocab.word2idx[vocab.END_TOKEN]
        line_id  = vocab.word2idx.get(vocab.NEWLINE_TOKEN, end_id)

        for poem in poems:
            lines = poem.get("lines", [])
            poem_ids: List[int] = []

            for line in lines:
                if not line:
                    continue
                line_ids = vocab.encode(line)
                if not line_ids:
                    continue
                poem_ids.append(start_id)
                poem_ids.extend(line_ids)
                poem_ids.append(end_id)
                poem_ids.append(line_id)

            # Strip trailing <LINE>
            while poem_ids and poem_ids[-1] == line_id:
                poem_ids.pop()

            if len(poem_ids) < 4:
                continue

            # Truncate to max_len - 1, then append final <END>
            if len(poem_ids) > self.max_len - 1:
                poem_ids = poem_ids[:self.max_len - 1]
            poem_ids.append(end_id)

            self.samples.append(poem_ids)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        ids = self.samples[idx]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
        }


def collate_poems(batch, pad_id: int):
    """Right-pad all sequences in the batch to the same length."""
    max_len = max(len(b["input_ids"]) for b in batch)
    B = len(batch)

    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    attn_mask = torch.zeros((B, max_len), dtype=torch.long)

    for i, b in enumerate(batch):
        seq = b["input_ids"]
        L = len(seq)
        input_ids[i, :L] = seq
        attn_mask[i, :L] = 1

    return {
        "input_ids": input_ids,
        "labels":    input_ids.clone(),   # target = same sequence (shifted in training loop)
        "attn_mask": attn_mask,
    }