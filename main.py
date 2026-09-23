import os
import argparse
import time
from typing import Optional


from src.utils import (
    setup_utf8_output,
    Colors,
    print_banner,
    print_section,
    save_poem_to_file,
    save_poem_to_json,
)

setup_utf8_output()


from src.preprocessing import BengaliPreprocessor
from src.vocabulary import Vocabulary
from src.word2vec_model import BengaliWord2Vec
from src.trigram_model import TrigramLanguageModel

from src.lstm_model import (
    BanglaLSTMTrainer,
    BanglaLSTMModel,
    TORCH_AVAILABLE,
)

from src.lstm_generator import LSTMPoemGenerator


DATA_PATH = os.path.join("data", "SAHITTO.ods")

MODELS_DIR = "models"

VOCAB_PATH = os.path.join(MODELS_DIR, "vocab.pkl")
LM_PATH = os.path.join(MODELS_DIR, "trigram_counts.pkl")
W2V_PATH = os.path.join(MODELS_DIR, "word2vec.model")
LSTM_PATH = os.path.join(MODELS_DIR, "lstm_model.pt")

OUTPUT_DIR = "output"


def _display_poem(result: dict, label: str = "LSTM") -> None:
    print()
    print(f"{Colors.BOLD}{Colors.YELLOW}" + "=" * 70 + f"{Colors.ENDC}")

    keyword = result.get("keyword", result.get("label", "Generated"))
    print(f"{Colors.BOLD}{Colors.YELLOW}"
          f"  📖 কবিতা: '{keyword}' [{label}]"
          f"{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.YELLOW}" + "=" * 70 + f"{Colors.ENDC}")

    lines = result.get("lines", [])
    if isinstance(lines, str):
        lines = lines.splitlines()
    if not lines:
        poem_text = result.get("poem_text", "")
        if poem_text:
            lines = poem_text.splitlines()

    for line in lines:
        line = str(line).strip()
        if line:
            print(line)
    print()

    # FIXED: correct metric keys
    metrics = result.get("metrics", {})
    if isinstance(metrics, dict):
        if "type_token_ratio" in metrics:
            print(f"শব্দ বৈচিত্র্য (TTR): {metrics['type_token_ratio']}")
        if "average_perplexity" in metrics:
            print(f"গড় Perplexity: {metrics['average_perplexity']}")
        if "rhyme_cadence_score" in metrics:
            print(f"অন্ত্যমিল স্কোর: {metrics['rhyme_cadence_score']}")

    scores = result.get("scoring_details", {})
    if isinstance(scores, dict):
        if "fluency_log_prob" in scores:
            print(f"Fluency Log-Prob: {scores['fluency_log_prob']}")
        if "rhyme_score" in scores:
            print(f"Rhyme Score: {scores['rhyme_score']}")

    print(f"{Colors.BOLD}{Colors.YELLOW}" + "=" * 70 + f"{Colors.ENDC}")


class PipelineManager:

    def __init__(self, data_path: str = DATA_PATH, force_retrain: bool = False):
        self.data_path = data_path
        self.force_retrain = force_retrain

        self.preprocessor: Optional[BengaliPreprocessor] = None
        self.vocab: Optional[Vocabulary] = None
        self.w2v: Optional[BengaliWord2Vec] = None
        self.lm: Optional[TrigramLanguageModel] = None

        self.lstm_model: Optional[BanglaLSTMModel] = None
        self.lstm_generator: Optional[LSTMPoemGenerator] = None

    def initialize(self):
        os.makedirs(MODELS_DIR, exist_ok=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        self.preprocessor = BengaliPreprocessor(self.data_path)

        cached = (
            not self.force_retrain
            and os.path.exists(VOCAB_PATH)
            and os.path.exists(LM_PATH)
            and os.path.exists(W2V_PATH)
        )

        if cached:
            print(f"{Colors.GREEN}"
                  f"✓ Found pre-trained models in '{MODELS_DIR}/'. Loading cached models..."
                  f"{Colors.ENDC}")
            t0 = time.time()

            self.vocab = Vocabulary.load(VOCAB_PATH)
            self.lm = TrigramLanguageModel.load(LM_PATH, self.vocab)
            self.w2v = BengaliWord2Vec.load(W2V_PATH)

            print(f"{Colors.GREEN}"
                  f"✓ Models loaded successfully in {time.time() - t0:.2f}s."
                  f"{Colors.ENDC}")
        else:
            print(f"{Colors.YELLOW}"
                  f"Training traditional NLP models from scratch..."
                  f"{Colors.ENDC}")
            t0 = time.time()

            (corpus_lines, corpus_poems,
             line_end_counts) = self.preprocessor.process_corpus(remove_duplicates=True)
            self.preprocessor.load_extra_poems(
                extra_dir=r"E:\4-1\Poems\Bengali-Poem-Dataset\dataset",
                apply_label_map=True,
            )

            print("\nBuilding Vocabulary (min_freq=5)...")
            self.vocab = Vocabulary(min_freq=5)
            self.vocab.build_vocabulary(corpus_lines)
            self.vocab.save(VOCAB_PATH)

            print("\nTraining Word2Vec Model...")
            self.w2v = BengaliWord2Vec(vector_size=100, window=5, min_count=1,
                                        epochs=20, sg=1)
            self.w2v.train(corpus_lines)
            self.w2v.save(W2V_PATH)

            print("\nTraining Trigram Language Model...")
            self.lm = TrigramLanguageModel(self.vocab)
            self.lm.train(corpus_lines, line_end_counts=line_end_counts)
            self.lm.save(LM_PATH)

            print(f"\n{Colors.GREEN}"
                  f"✓ Traditional NLP pipeline completed in {time.time() - t0:.2f}s."
                  f"{Colors.ENDC}")

        # Baseline LSTM retraining
        if self.force_retrain and TORCH_AVAILABLE:
            try:
                print(f"\n{Colors.YELLOW}"
                      f"Retraining baseline LSTM with fixed settings..."
                      f"{Colors.ENDC}")

                trainer = BanglaLSTMTrainer(
                    vocab=self.vocab,
                    seq_len=32,
                    embed_dim=128,
                    hidden_dim=256,
                    num_layers=2,
                    dropout=0.4,
                    batch_size=64,
                    epochs=25,
                    lr=0.001,
                    patience=4,
                )
                trainer.train(self.preprocessor.corpus_lines)
                trainer.save(LSTM_PATH)

                print(f"{Colors.GREEN}"
                      f"✓ Baseline LSTM retrained and saved to {LSTM_PATH}"
                      f"{Colors.ENDC}")
            except Exception as e:
                print(f"{Colors.RED}"
                      f"✗ Baseline LSTM retraining failed: {e}"
                      f"{Colors.ENDC}")

        # LSTM Models
        self._initialize_lstm()

    def _initialize_lstm(self):
        if not TORCH_AVAILABLE:
            print(f"{Colors.YELLOW}"
                  f"⚠ PyTorch not found — LSTM generators disabled."
                  f"{Colors.ENDC}")
            return

        if (not self.force_retrain and os.path.exists(LSTM_PATH)):
            print(f"{Colors.GREEN}"
                  f"✓ Loading cached LSTM model from '{LSTM_PATH}'..."
                  f"{Colors.ENDC}")
            try:
                t0 = time.time()
                self.lstm_model = BanglaLSTMTrainer.load_model(LSTM_PATH, self.vocab)
                self.lstm_generator = LSTMPoemGenerator(
                    model=self.lstm_model,
                    vocab=self.vocab,
                    w2v=self.w2v,
                    line_end_counts=self.lm.line_end_counts,
                    line_start_counts=self.lm.line_start_counts,
                )
                print(f"{Colors.GREEN}"
                      f"✓ Baseline LSTM loaded in {time.time() - t0:.2f}s."
                      f"{Colors.ENDC}")
            except Exception as e:
                print(f"{Colors.RED}"
                      f"✗ Failed to load baseline LSTM: {e}"
                      f"{Colors.ENDC}")
                self.lstm_model = None
                self.lstm_generator = None

        elif self.force_retrain:
            if os.path.exists(LSTM_PATH):
                try:
                    self.lstm_model = BanglaLSTMTrainer.load_model(LSTM_PATH, self.vocab)
                    self.lstm_generator = LSTMPoemGenerator(
                        model=self.lstm_model,
                        vocab=self.vocab,
                        w2v=self.w2v,
                        line_end_counts=self.lm.line_end_counts,
                        line_start_counts=self.lm.line_start_counts,
                    )
                except Exception as e:
                    print(f"{Colors.RED}"
                          f"✗ Failed to reload baseline LSTM: {e}"
                          f"{Colors.ENDC}")
        else:
            if not os.path.exists(LSTM_PATH):
                print(f"{Colors.YELLOW}"
                      f"⚠ Baseline LSTM checkpoint not found: {LSTM_PATH}"
                      f"{Colors.ENDC}")


def interactive_cli(pipeline: PipelineManager):
    while True:
        print()
        print_banner("Bangla Poetry Generator")

        lstm_status = (
            f"{Colors.GREEN}[ready]{Colors.ENDC}"
            if pipeline.lstm_generator else f"{Colors.RED}[unavailable]{Colors.ENDC}")

        print(f"  1. 🧠 Generate Poem — LSTM {lstm_status}")
        print("  2. ℹ️ Model Information")
        print("  3. 🚪 Exit")
        print()

        choice = input("Select an option [1-3]: ").strip()

        if choice == "1":
            if not pipeline.lstm_generator:
                print(f"{Colors.RED}✗ LSTM generator unavailable.{Colors.ENDC}")
                continue

            print_section("Poem Generation — LSTM")
            keyword = input("Enter keyword: ").strip()
            if not keyword:
                print(f"{Colors.YELLOW}⚠ Keyword cannot be empty.{Colors.ENDC}")
                continue

            try:
                lines = int(input("Number of lines [4]: ").strip() or "4")
                temperature = float(input("Temperature [0.85]: ").strip() or "0.85")
                result = pipeline.lstm_generator.generate_poem(
                    keyword=keyword,
                    num_lines=max(1, min(lines, 6)),
                    num_candidates=5,
                    min_words_per_line=5,
                    max_words_per_line=8,
                    top_k=20,
                    temperature=temperature,
                    top_p=0.90,
                )
                _display_poem(result, label="LSTM")

                # Save
                try:
                    save_poem_to_file(
                        result["poem_text"],
                        keyword=keyword,
                        metadata={
                            "lines": result["num_lines"],
                            "avg_perplexity": result["metrics"].get("average_perplexity"),
                            "ttr": result["metrics"].get("type_token_ratio"),
                        },
                    )
                    save_poem_to_json(result["lines"], keyword=keyword)
                except Exception:
                    pass

            except Exception as e:
                print(f"{Colors.RED}✗ Generation failed: {e}{Colors.ENDC}")

        elif choice == "2":
            print_section("Model Information")
            print(f"LSTM: {'Ready' if pipeline.lstm_generator else 'Unavailable'}")
            print()
            print(f"Vocabulary  : {VOCAB_PATH}")
            print(f"Trigram LM  : {LM_PATH}")
            print(f"Word2Vec    : {W2V_PATH}")
            print(f"LSTM model  : {LSTM_PATH}")
            print()
            print(f"Vocab size  : {pipeline.vocab.vocab_size if pipeline.vocab else 'N/A'}")

        elif choice == "3":
            print(f"{Colors.GREEN}✓ Goodbye!{Colors.ENDC}")
            break
        else:
            print(f"{Colors.YELLOW}⚠ Invalid option.{Colors.ENDC}")


def main():
    parser = argparse.ArgumentParser(
        description="Bangla Poetry Generator — LSTM"
    )

    parser.add_argument("--keyword", type=str, default=None)
    parser.add_argument("--lines", type=int, default=4)
    parser.add_argument("--temp", type=float, default=0.85)
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--top_p", type=float, default=0.90)
    parser.add_argument("--rhyme-scheme", type=str, default="auto", choices=["AABB", "auto", "none"])

    args = parser.parse_args()

    if args.lines < 1:
        args.lines = 1
    if args.lines > 6:
        args.lines = 6
    if args.temp <= 0:
        print(f"{Colors.RED}✗ Temperature must be greater than 0.{Colors.ENDC}")
        return

    pipeline = PipelineManager(force_retrain=args.retrain)

    try:
        pipeline.initialize()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}⚠ Interrupted by user.{Colors.ENDC}")
        return
    except Exception as e:
        print(f"\n{Colors.RED}✗ Pipeline initialization failed: {e}{Colors.ENDC}")
        return

    if args.keyword:
        if not pipeline.lstm_generator:
            print(f"{Colors.RED}"
                  f"✗ LSTM generator is not available."
                  f"{Colors.ENDC}")
            return
        try:
            result = pipeline.lstm_generator.generate_poem(
                keyword=args.keyword,
                num_lines=args.lines,
                num_candidates=5,
                min_words_per_line=5,
                max_words_per_line=8,
                top_k=args.top_k,
                temperature=args.temp,
                top_p=args.top_p,
                rhyme_scheme=args.rhyme_scheme,
            )
            _display_poem(result, label="LSTM")

            # Save poem
            try:
                save_poem_to_file(
                    result["poem_text"],
                    keyword=args.keyword,
                    metadata={
                        "lines": result["num_lines"],
                        "avg_perplexity": result["metrics"].get("average_perplexity"),
                        "ttr": result["metrics"].get("type_token_ratio"),
                    },
                )
                save_poem_to_json(result["lines"], keyword=args.keyword)
            except Exception:
                pass

        except Exception as e:
            print(f"{Colors.RED}✗ LSTM generation failed: {e}{Colors.ENDC}")
        return

    interactive_cli(pipeline)


if __name__ == "__main__":
    main()