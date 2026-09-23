"""
src/lstm_label_generator.py
----------------------------
Label-Conditioned LSTM Bengali Poetry Generator.

FIXED:
- No hard-coded label anchor lists (labels influence via neural embedding only)
- Gentler sampling: temp=0.85, top_k=20, top_p=0.90
- Correct <LINE> hidden-state passing between lines
- Mild repetition penalties
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, Set

import torch
import torch.nn.functional as F

from src.lstm_label_model import LabelConditionedBanglaLSTMModel
from src.preprocessing import BENGALI_STOP_WORDS
from src.utils import (
    Colors,
    compute_poem_statistics,
    print_banner,
    print_section,
    save_poem_to_file,
    save_poem_to_json,
    setup_utf8_output,
)
from src.vocabulary import Vocabulary

setup_utf8_output()

DEFAULT_CHECKPOINT_PATH = os.path.join("models", "lstm_label_model.pt")
DEFAULT_VOCAB_PATH = os.path.join("models", "vocab.pkl")
DEFAULT_OUTPUT_DIR = "output"


class LabelConditionedLSTMPoemGenerator:
    """
    Generates Bengali poetry using a trained LabelConditionedBanglaLSTMModel.
    Label conditioning is purely neural — no hard-coded anchor lists.
    """

    def __init__(
        self,
        checkpoint_path: str = DEFAULT_CHECKPOINT_PATH,
        vocab_path: str = DEFAULT_VOCAB_PATH,
        device: Optional[torch.device] = None,
    ):
        self.checkpoint_path = checkpoint_path
        self.vocab_path = vocab_path
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.vocab: Vocabulary
        self.model: LabelConditionedBanglaLSTMModel
        self.label2idx: Dict[str, int]
        self.idx2label: Dict[int, str]
        self.lstm_config: Dict[str, Any]

        self._load_and_verify()

    def _load_and_verify(self) -> None:
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at '{self.checkpoint_path}'.")
        if not os.path.exists(self.vocab_path):
            raise FileNotFoundError(f"Vocabulary not found at '{self.vocab_path}'.")

        self.vocab = Vocabulary.load(self.vocab_path)

        expected_specials = {
            "<PAD>": 0, "<UNK>": 1, "<START>": 2, "<END>": 3, "<LINE>": 4,
        }
        for token, expected_id in expected_specials.items():
            actual_id = self.vocab.word2idx.get(token)
            if actual_id != expected_id:
                raise ValueError(
                    f"Vocabulary incompatibility: token '{token}' has id {actual_id}, "
                    f"expected {expected_id}."
                )

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device,
                                weights_only=False)

        for key in ("lstm_config", "state_dict", "label2idx"):
            if key not in checkpoint:
                raise KeyError(f"Checkpoint missing '{key}'.")

        self.lstm_config = checkpoint["lstm_config"]
        self.label2idx = checkpoint["label2idx"]
        self.idx2label = checkpoint.get("idx2label",
                                        {i: l for l, i in self.label2idx.items()})

        if self.lstm_config.get("vocab_size") != self.vocab.vocab_size:
            raise ValueError("Vocabulary size mismatch with checkpoint.")

        self.model = LabelConditionedBanglaLSTMModel(
            vocab_size=self.lstm_config["vocab_size"],
            num_labels=self.lstm_config["num_labels"],
            embed_dim=self.lstm_config["embed_dim"],
            label_embed_dim=self.lstm_config["label_embed_dim"],
            hidden_dim=self.lstm_config["hidden_dim"],
            num_layers=self.lstm_config["num_layers"],
            dropout=self.lstm_config.get("dropout", 0.5),
            pad_idx=self.lstm_config.get("pad_idx", 0),
        )
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.stopword_ids: Set[int] = {
            self.vocab.word2idx[w] for w in BENGALI_STOP_WORDS
            if w in self.vocab.word2idx
        }

    def resolve_label(self, label_input) -> Tuple[str, int]:
        if isinstance(label_input, int):
            if label_input not in self.idx2label:
                raise ValueError(f"Invalid label id {label_input}")
            return self.idx2label[label_input], label_input
        label_str = str(label_input).strip()
        if label_str in self.label2idx:
            return label_str, self.label2idx[label_str]
        for name, idx in self.label2idx.items():
            if name.lower() == label_str.lower():
                return name, idx
        available = ", ".join(f"'{k}' (ID {v})" for k, v in self.label2idx.items())
        raise ValueError(f"Unknown label '{label_input}'.\nAvailable: {available}")

    def _is_line_acceptable(
        self,
        token_ids: List[int],
        min_unique: int = 3,
        max_repeat_ratio: float = 0.55,
    ) -> bool:
        if not token_ids:
            return False
        if len(set(token_ids)) < min_unique:
            return False
        most_common = Counter(token_ids).most_common(1)[0][1]
        if most_common / len(token_ids) > max_repeat_ratio:
            return False
        if self.stopword_ids:
            has_content = any(tid not in self.stopword_ids for tid in token_ids)
            if not has_content:
                return False
        return True

    def _sample_next_token(
        self,
        logits: torch.Tensor,
        current_line_ids: List[int],
        generated_token_ids: List[int],
        min_words_per_line: int,
        temperature: float = 0.85,
        top_k: int = 20,
        top_p: float = 0.90,
        repetition_penalty: float = 1.08,
        greedy: bool = False,
        debug: bool = False,
        step_idx: Optional[int] = None,
    ) -> Tuple[int, float]:
        logits = logits.clone()

        pad_id = self.vocab.word2idx["<PAD>"]
        unk_id = self.vocab.word2idx["<UNK>"]
        start_id = self.vocab.word2idx["<START>"]
        end_id = self.vocab.word2idx["<END>"]
        line_id = self.vocab.word2idx.get("<LINE>", end_id)

        for tid in (pad_id, unk_id, start_id, line_id):
            logits[tid] = -1e9

        if len(current_line_ids) < min_words_per_line:
            logits[end_id] = -1e9

        if current_line_ids:
            prev = current_line_ids[-1]
            if prev not in (end_id, start_id, pad_id, unk_id, line_id):
                logits[prev] -= 3.0

        if generated_token_ids:
            recent = set(generated_token_ids[-20:])
            for tid in recent:
                if tid not in (end_id, start_id, pad_id, unk_id, line_id):
                    logits[tid] -= 0.8

        if self.stopword_ids:
            consecutive = 0
            for prev_tid in reversed(current_line_ids):
                if prev_tid in self.stopword_ids:
                    consecutive += 1
                else:
                    break
            if consecutive >= 3:
                for sid in self.stopword_ids:
                    if sid not in (end_id, start_id, pad_id, unk_id, line_id):
                        logits[sid] -= 1.5

        if repetition_penalty > 1.0 and generated_token_ids:
            recent = generated_token_ids[-50:]
            counts = Counter(recent)
            for tid, count in counts.items():
                if tid in (end_id, start_id, pad_id, unk_id, line_id):
                    continue
                penalty = repetition_penalty ** count
                if logits[tid] > 0:
                    logits[tid] /= penalty
                else:
                    logits[tid] *= penalty

        if debug and step_idx is not None and step_idx < 5:
            probs_dbg = F.softmax(logits / max(temperature, 0.05), dim=-1)
            top10 = torch.topk(probs_dbg, min(10, probs_dbg.size(-1)))
            print(f"  [Step {step_idx}] Top candidates:")
            for val, idx in zip(top10.values.tolist(), top10.indices.tolist()):
                print(f"    • {self.vocab.idx_to_word(idx):<18} (ID {idx:>5}) : p={val:.4f}")

        logits = logits / max(temperature, 0.05)

        if top_k > 0:
            kth = torch.topk(logits, min(top_k, logits.size(-1))).values[-1]
            logits[logits < kth] = -1e9

        probs = F.softmax(logits, dim=-1)

        if top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            remove_mask = (cumulative - sorted_probs) > top_p
            sorted_probs[remove_mask] = 0.0
            probs = torch.zeros_like(probs).scatter_(-1, sorted_idx, sorted_probs)
            total = probs.sum()
            if total > 0:
                probs = probs / total
            else:
                probs = F.softmax(logits, dim=-1)

        if greedy:
            chosen_id = int(torch.argmax(probs).item())
        else:
            chosen_id = int(torch.multinomial(probs, num_samples=1).item())
        chosen_prob = float(probs[chosen_id].item())

        return chosen_id, chosen_prob

    @torch.no_grad()
    def _generate_one_line(
        self,
        label_tensor: torch.Tensor,
        hidden,
        prompt_tokens: List[str],
        prompt_token_ids: List[int],
        all_generated_token_ids: List[int],
        is_first_line: bool,
        min_words_per_line: int,
        max_words_per_line: int,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        greedy: bool,
        debug: bool,
        start_step_idx: int,
        max_retries: int = 3,
        min_unique_tokens: int = 3,
        max_repeat_ratio: float = 0.55,
    ) -> Tuple[str, List[int], Any, float, int]:
        start_id = self.vocab.word2idx["<START>"]
        end_id = self.vocab.word2idx["<END>"]

        best_line, best_ids, best_hidden, best_logp, best_steps = "", [], hidden, 0.0, 0

        for attempt in range(max_retries):
            if is_first_line and attempt == 0:
                warmup_ids = [start_id] + prompt_token_ids
                x_prime = torch.tensor([warmup_ids], dtype=torch.long, device=self.device)
                logits_prime, attempt_hidden = self.model(x_prime, label_tensor, hidden)
                last_logit = logits_prime[0, -1, :]
            else:
                x_prime = torch.tensor([[start_id]], dtype=torch.long, device=self.device)
                logits_prime, attempt_hidden = self.model(x_prime, label_tensor, hidden)
                last_logit = logits_prime[0, -1, :]

            line_tokens: List[str] = list(prompt_tokens) if (is_first_line and attempt == 0) else []
            line_ids: List[int] = list(prompt_token_ids) if (is_first_line and attempt == 0) else []
            attempt_all_ids: List[int] = list(all_generated_token_ids)
            total_logp = 0.0
            step_idx = start_step_idx

            while True:
                next_id, next_prob = self._sample_next_token(
                    logits=last_logit,
                    current_line_ids=line_ids,
                    generated_token_ids=attempt_all_ids,
                    min_words_per_line=min_words_per_line,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    greedy=greedy,
                    debug=debug and (attempt == 0),
                    step_idx=step_idx,
                )
                step_idx += 1
                total_logp += math.log(max(next_prob, 1e-12))

                is_end = (next_id == end_id) or (len(line_ids) >= max_words_per_line)
                if is_end:
                    break

                word = self.vocab.idx_to_word(next_id)
                line_tokens.append(word)
                line_ids.append(next_id)
                attempt_all_ids.append(next_id)

                x_step = torch.tensor([[next_id]], dtype=torch.long, device=self.device)
                logits_step, attempt_hidden = self.model(x_step, label_tensor, attempt_hidden)
                last_logit = logits_step[0, -1, :]

            line_str = " ".join(line_tokens).strip()

            if self._is_line_acceptable(line_ids, min_unique_tokens, max_repeat_ratio):
                return line_str, line_ids, attempt_hidden, total_logp, step_idx - start_step_idx

            if attempt == 0 or len(set(line_ids)) > len(set(best_ids)):
                best_line = line_str
                best_ids = line_ids
                best_hidden = attempt_hidden
                best_logp = total_logp
                best_steps = step_idx - start_step_idx

        return best_line, best_ids, best_hidden, best_logp, best_steps

    @torch.no_grad()
    def generate_poem(
        self,
        label: str | int = "Separation",
        prompt: Optional[str] = None,
        num_lines: int = 4,
        min_words_per_line: int = 4,
        max_words_per_line: int = 8,
        temperature: float = 0.85,
        top_k: int = 20,
        top_p: float = 0.90,
        repetition_penalty: float = 1.08,
        greedy: bool = False,
        debug: bool = False,
    ) -> Dict[str, Any]:
        num_lines = max(4, min(num_lines, 6))
        label_name, label_id = self.resolve_label(label)

        prompt_display = f"'{prompt.strip()}'" if (prompt and prompt.strip()) else "(None)"
        print(f"\n{Colors.CYAN}{'=' * 65}{Colors.ENDC}")
        print(f"{Colors.BOLD}{Colors.YELLOW}  🤖 Label-Conditioned LSTM Poetry Generator{Colors.ENDC}")
        print(f"{Colors.CYAN}{'=' * 65}{Colors.ENDC}")
        print(f"  • Label                : {Colors.GREEN}{label_name}{Colors.ENDC}")
        print(f"  • Label ID             : {label_id}")
        print(f"  • Checkpoint           : {self.checkpoint_path}")
        print(f"  • Vocab size           : {self.vocab.vocab_size:,}")
        print(f"  • Prompt               : {prompt_display}")
        print(f"  • Temp / Top-k / Top-p : {temperature} / {top_k} / {top_p}")
        print(f"  • Repetition Penalty   : {repetition_penalty}")
        print(f"{Colors.CYAN}{'=' * 65}{Colors.ENDC}\n")

        start_id = self.vocab.word2idx["<START>"]
        end_id = self.vocab.word2idx["<END>"]
        line_id = self.vocab.word2idx.get("<LINE>", end_id)
        label_tensor = torch.tensor([label_id], dtype=torch.long, device=self.device)

        prompt_tokens: List[str] = []
        prompt_token_ids: List[int] = []
        if prompt and prompt.strip():
            for raw_w in prompt.strip().split():
                w = raw_w.strip()
                if not w:
                    continue
                wid = self.vocab.word_to_idx(w)
                prompt_tokens.append(w)
                prompt_token_ids.append(wid)

        x_init = torch.tensor([[start_id]], dtype=torch.long, device=self.device)
        _, hidden = self.model(x_init, label_tensor)

        lines: List[str] = []
        all_generated_token_ids: List[int] = list(prompt_token_ids)
        total_log_prob = 0.0
        sampled_token_count = 0
        step_idx = 0

        for line_num in range(num_lines):
            is_first = (line_num == 0)
            line_str, line_ids, hidden, line_logp, steps = self._generate_one_line(
                label_tensor=label_tensor,
                hidden=hidden,
                prompt_tokens=prompt_tokens if is_first else [],
                prompt_token_ids=prompt_token_ids if is_first else [],
                all_generated_token_ids=all_generated_token_ids,
                is_first_line=is_first,
                min_words_per_line=min_words_per_line,
                max_words_per_line=max_words_per_line,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                greedy=greedy,
                debug=debug,
                start_step_idx=step_idx,
            )
            if line_str:
                lines.append(line_str)
            all_generated_token_ids.extend(line_ids)
            total_log_prob += line_logp
            sampled_token_count += steps
            step_idx += steps

            if line_num < num_lines - 1:
                trans = torch.tensor([[end_id, line_id]], dtype=torch.long, device=self.device)
                _, hidden = self.model(trans, label_tensor, hidden)

        formatted_lines = []
        for i, l in enumerate(lines):
            clean_l = l.strip(",|। \t")
            if i == len(lines) - 1:
                formatted_lines.append(f"{clean_l}।")
            else:
                formatted_lines.append(f"{clean_l},")
        poem_text = "\n".join(formatted_lines)

        stats = compute_poem_statistics(formatted_lines)
        avg_nll = -total_log_prob / max(1, sampled_token_count)
        avg_perplexity = round(math.exp(min(avg_nll, 20)), 2)

        return {
            "label": label_name,
            "label_id": label_id,
            "prompt": prompt or "",
            "lines": formatted_lines,
            "poem_text": poem_text,
            "metrics": {
                "total_words": stats["total_words"],
                "avg_words_per_line": stats["avg_words_per_line"],
                "ttr": stats["lexical_richness_ttr"],
                "average_perplexity": avg_perplexity,
            },
            "parameters": {
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty,
                "min_words_per_line": min_words_per_line,
                "max_words_per_line": max_words_per_line,
            },
        }


def display_poem(result: Dict[str, Any]) -> None:
    label = result["label"]
    label_id = result["label_id"]
    metrics = result["metrics"]

    print(f"\n{Colors.CYAN}╭─────────────────────────────────────────────────────────────╮{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.YELLOW}  📖  কবিতা: '{label}' (ID: {label_id})  [Label-LSTM]{Colors.ENDC}")
    print(f"{Colors.CYAN}├─────────────────────────────────────────────────────────────┤{Colors.ENDC}")
    for line in result["lines"]:
        print(f"  {Colors.BOLD}{line}{Colors.ENDC}")
    print(f"{Colors.CYAN}╰─────────────────────────────────────────────────────────────╯{Colors.ENDC}\n")

    print(f"{Colors.CYAN}--- Generation & Linguistic Diagnostics ---{Colors.ENDC}")
    print(f"• Total Words         : {metrics['total_words']}")
    print(f"• Words per Line      : {metrics['avg_words_per_line']}")
    print(f"• Lexical Diversity   : {metrics['ttr']:.4f} (Type-Token Ratio)")
    print(f"• Average Perplexity  : {metrics['average_perplexity']}")
    print(f"• Prompt Used         : {result['prompt'] if result['prompt'] else '(None)'}")


def main():
    parser = argparse.ArgumentParser(description="Label-Conditioned LSTM Bengali Poetry")
    parser.add_argument("--label", type=str, default="Separation")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--lines", type=int, default=4)
    parser.add_argument("--temp", type=float, default=0.85)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--top_p", type=float, default=0.90)
    parser.add_argument("--rep_penalty", type=float, default=1.08)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    generator = LabelConditionedLSTMPoemGenerator()
    result = generator.generate_poem(
        label=args.label,
        prompt=args.prompt,
        num_lines=args.lines,
        temperature=args.temp,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.rep_penalty,
        greedy=args.greedy,
        debug=args.debug,
    )
    display_poem(result)

    save_poem_to_file(
        result["poem_text"],
        keyword=f"Label_{result['label']}",
        metadata={
            "model": "Label-Conditioned LSTM",
            "label": result["label"],
            "label_id": result["label_id"],
            "prompt": result["prompt"],
            "ttr": result["metrics"]["ttr"],
            "avg_perplexity": result["metrics"]["average_perplexity"],
            "temperature": args.temp,
        },
    )
    save_poem_to_json(result["lines"], keyword=f"Label_{result['label']}")
    print(f"\n{Colors.GREEN}✓ Poem saved to output/generated_poems.txt and output/poems.json{Colors.ENDC}\n")


if __name__ == "__main__":
    main()