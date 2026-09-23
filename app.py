import os
import sys
import json
import time
import pickle
import random
import threading
from collections import Counter

from flask import Flask, render_template, request, jsonify

try:
    import torch
except ImportError:          # the LSTM needs torch anyway; this only guards the import
    torch = None

# Ensure src imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.vocabulary import Vocabulary
from src.word2vec_model import BengaliWord2Vec
from src.lstm_model import BanglaLSTMTrainer
from src.lstm_generator import LSTMPoemGenerator
from src.trigram_model import TrigramLanguageModel
from src.smoothing import LaplaceSmoothing
from src.generator import BanglaPoemGenerator

try:
    from src.transformer_model import BanglaTransformerTrainer
    from src.transformer_generator import TransformerPoemGenerator
    TRANSFORMER_AVAILABLE = True
except ImportError:
    TRANSFORMER_AVAILABLE = False

app = Flask(__name__, template_folder="templates", static_folder="static")

MODELS_DIR = "models"
VOCAB_PATH = os.path.join(MODELS_DIR, "vocab_min5.pkl")
LM_PATH = os.path.join(MODELS_DIR, "trigram_counts.pkl")
W2V_PATH = os.path.join(MODELS_DIR, "word2vec.model")
LSTM_PATH = os.path.join(MODELS_DIR, "lstm_model_topic.pt")
TRANSFORMER_PATH = os.path.join(MODELS_DIR, "transformer_model.pt")
LINE_COUNTS_PATH = os.path.join(MODELS_DIR, "line_counts_min5.pkl")
TOPIC_STATS_PATH = os.path.join(MODELS_DIR, "topic_stats_min5.pkl")


# Global model holders (loaded once)

_state = {
    "vocab": None,
    "w2v": None,
    "lm": None,
    "smoother": None,
    "lstm_gen": None,
    "transformer_gen": None,
    "trigram_gen": None,
}

# global RNG state + shared models are not safe to use from several request threads at once
_gen_lock = threading.Lock()


def _load_line_counts():
    """(line_start_counts, line_end_counts) saved at preprocessing time, or empty Counters."""
    if os.path.exists(LINE_COUNTS_PATH):
        with open(LINE_COUNTS_PATH, "rb") as f:
            d = pickle.load(f)
        return d.get("line_start_counts", Counter()), d.get("line_end_counts", Counter())
    return Counter(), Counter()


def load_all_models():
    """Load all models once at server startup."""
    print("\n" + "=" * 60)
    print("  Loading models for web interface...")
    print("=" * 60)

    # Vocab
    print(f"  Loading vocab from {VOCAB_PATH}...")
    _state["vocab"] = Vocabulary.load(VOCAB_PATH)
    print(f"   Vocab: {_state['vocab'].vocab_size} tokens")

    # Word2Vec
    if os.path.exists(W2V_PATH):
        print("  Loading Word2Vec...")
        _state["w2v"] = BengaliWord2Vec.load(W2V_PATH)
        print(f"   W2V: {_state['w2v'].vocab_size()} words")
        if _state["w2v"].vocab_size() > _state["vocab"].vocab_size * 1.2:
            print("  ! Word2Vec vocab is much larger than the LSTM vocab. Were they trained on the "
                  "same corpus? Theme words missing from the LSTM vocab are ignored.")

    # Trigram + smoothing (for trigram backend)
    if os.path.exists(LM_PATH):
        print("  Loading Trigram LM...")
        _state["lm"] = TrigramLanguageModel.load(LM_PATH, _state["vocab"])
        _state["smoother"] = LaplaceSmoothing(_state["lm"], alpha=1.0)
        _state["trigram_gen"] = BanglaPoemGenerator(
            lm=_state["lm"],
            smoother=_state["smoother"],
            w2v=_state["w2v"],
            vocab=_state["vocab"],
        )
        print("  Trigram ready")

    # LSTM
    if os.path.exists(LSTM_PATH):
        print("  Loading LSTM...")
        lstm_model = BanglaLSTMTrainer.load_model(LSTM_PATH, _state["vocab"])

        start_counts, end_counts = _load_line_counts()
        lm = _state["lm"]
        if not end_counts and lm is not None:
            end_counts = getattr(lm, "line_end_counts", Counter())
        if not start_counts and lm is not None:
            start_counts = getattr(lm, "line_start_counts", Counter())
        if not start_counts:
            print(f"  ! No line-start counts found ({LINE_COUNTS_PATH}). Run "
                  "BengaliPreprocessor.save_line_counts() after preprocessing.")
        if not end_counts:
            print("  ! No line-end counts found: rhyme control will be weak.")

        _state["lstm_gen"] = LSTMPoemGenerator(
            model=lstm_model,
            vocab=_state["vocab"],
            w2v=_state["w2v"],
            line_end_counts=end_counts,
            line_start_counts=start_counts,
        )
        print(f"  ✓ LSTM ready (end_mode={_state['lstm_gen'].end_mode}, "
              f"masked junk tokens={_state['lstm_gen'].n_masked_junk})")

    # Transformer (optional)
    if TRANSFORMER_AVAILABLE and os.path.exists(TRANSFORMER_PATH):
        try:
            print("  Loading Transformer...")
            t_model = BanglaTransformerTrainer.load_model(
                TRANSFORMER_PATH, _state["vocab"]
            )
            _state["transformer_gen"] = TransformerPoemGenerator(
                model=t_model,
                vocab=_state["vocab"],
            )
            print("   Transformer ready (keyword-based, no labels)")
        except Exception as e:
            print(f"   Transformer load failed: {e}")

    print("=" * 60)
    print("  All models loaded. Server ready.")
    print("=" * 60 + "\n")


# Helpers

def _as_bool(value, default):
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _clamp(value, lo, hi):
    return max(lo, min(value, hi))


def _jsonable(obj):
    """Make nested result dicts safe for jsonify (sets, Counters, numpy/torch scalars ...)."""
    return json.loads(json.dumps(obj, default=str))


def _seed_everything(seed):
    try:
        seed = int(seed)
    except (TypeError, ValueError):
        return
    random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)



# Routes

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/models")
def api_models():
    """Returns list of available models."""
    models = []
    if _state["lstm_gen"]:
        models.append({"id": "lstm", "label": "Baseline LSTM (Recommended)"})
    if _state["transformer_gen"]:
        models.append({"id": "transformer", "label": "Transformer (Keyword-based)"})
    if _state["trigram_gen"]:
        models.append({"id": "trigram", "label": "Trigram LM (Traditional NLP)"})

    return jsonify({
        "models": models,
        "transformer_labels": [],   # no labels — transformer now uses keyword like LSTM
    })


@app.route("/api/generate", methods=["POST"])
def api_generate():
    """Generate a poem."""
    try:
        data = request.get_json(force=True)
        model_id = data.get("model", "lstm")
        keyword = (data.get("keyword") or "").strip()
        num_lines = _clamp(int(data.get("num_lines", 4)), 2, 6)
        temperature = float(data.get("temperature", 0.85))
        top_k = int(data.get("top_k", 20))
        top_p = float(data.get("top_p", 0.90))
        seed = data.get("seed")

        # LSTM-only options (handy for ablation studies)
        num_candidates = _clamp(int(data.get("num_candidates", 5)), 1, 12)
        use_beam_search = _as_bool(data.get("use_beam_search"), False)
        use_enhancer = _as_bool(data.get("use_enhancer"), True)
        end_mode = data.get("end_mode")            # None | "model" | "heuristic"
        if end_mode not in (None, "", "model", "heuristic"):
            return jsonify({"error": "end_mode must be 'model' or 'heuristic'"}), 400

        rhyme_scheme = data.get("rhyme_scheme", "auto")
        if rhyme_scheme not in ("AABB", "auto", "none"):
            return jsonify({"error": "rhyme_scheme must be AABB / auto / none"}), 400

        if not keyword:
            return jsonify({"error": "Keyword is required"}), 400

        t0 = time.time()

        with _gen_lock:
            _seed_everything(seed)

            # Dispatch to correct generator Ekhan theke start main model
            if model_id == "lstm":
                if not _state["lstm_gen"]:
                    return jsonify({"error": "LSTM model not loaded"}), 500
                result = _state["lstm_gen"].generate_poem(
                    keyword=keyword,
                    num_lines=num_lines,
                    num_candidates=num_candidates,
                    min_words_per_line=5,
                    max_words_per_line=8,
                    top_k=top_k,
                    temperature=temperature,
                    top_p=top_p,
                    use_beam_search=use_beam_search,
                    use_enhancer=use_enhancer,
                    end_mode=end_mode or None,
                    save_html=False,
                    rhyme_scheme=rhyme_scheme,
                )
                label_info = None

            elif model_id == "transformer":
                gen = _state["transformer_gen"]
                if not gen:
                    return jsonify({"error": "Transformer model not loaded"}), 500
                result = gen.generate_poem(
                    keyword=keyword,
                    num_lines=num_lines,
                    temperature=temperature,
                    top_k=top_k if top_k > 0 else 30,
                    top_p=top_p if top_p > 0 else 0.92,
                    rep_penalty=float(data.get("rep_penalty", 1.15)),
                )
                label_info = None

            elif model_id == "trigram":
                if not _state["trigram_gen"]:
                    return jsonify({"error": "Trigram model not loaded"}), 500
                result = _state["trigram_gen"].generate_poem(
                    keyword=keyword,
                    num_lines=num_lines,
                    num_candidates=5,
                    min_words_per_line=5,
                    max_words_per_line=8,
                    top_k=5,
                    temperature=temperature,
                )
                label_info = None

            else:
                return jsonify({"error": f"Unknown model: {model_id}"}), 400

        elapsed = round(time.time() - t0, 2)

        return jsonify({
            "success": True,
            "model": model_id,
            "keyword": keyword,
            "poem_text": result.get("poem_text", ""),
            "lines": result.get("lines", []),
            "metrics": _jsonable(result.get("metrics", {})),
            "scoring_details": _jsonable(result.get("scoring_details")),
            "enhancement": _jsonable(result.get("enhancement")),
            "thematic_coverage": result.get("thematic_coverage"),
            "end_mode": result.get("end_mode"),
            "label_info": label_info,
            "seed": seed,
            "elapsed_seconds": elapsed,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        })

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500



# Entry point

if __name__ == "__main__":
    load_all_models()
    print("\n  Open http://127.0.0.1:5000 in your browser\n")
    app.run(debug=False, host="127.0.0.1", port=5000)