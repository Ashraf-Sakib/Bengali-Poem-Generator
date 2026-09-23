# Bengali Poem Generator

A neural and statistical Bengali poetry generation system with LSTM language modeling, Word2Vec semantic embeddings, phonetic rhyme planning, and a trigram baseline.

**Course:** CSE 4122 - Natural Language Processing Laboratory
**Department:** CSE, Khulna University of Engineering & Technology (KUET)

## Features

- Generate 4-6 line Bengali poems from a single keyword
- Two LSTM backends: baseline and topic-conditioned
- Bengali phonetic rhyme planning with 21 poetic codas
- Constrained decoding: theme bias, stem repetition penalty, line-end validation, rhyme steering
- Best-of-N reranking for quality selection
- Flask web interface and CLI

## Setup

### 1. Install dependencies

```
install required dependencies 
```

### 2. Get trained models

Trained models are not included in this repo (file size). Download from the release page and place them in the `models/` folder:

```
models/
  vocab.pkl
  vocab_min5.pkl
  word2vec.model
  trigram_counts.pkl
  lstm_model.pt
  lstm_model_topic.pt
  line_counts.pkl
  line_counts_min5.pkl
  topic_stats.pkl
  topic_stats_min5.pkl
```

### 3. Get the dataset

Place `SAHITTO.ods` in the `data/` folder.

## Usage

### Web interface

```
python app.py
```

Open http://127.0.0.1:5000 in the browser.

Switch between models using environment variables.

Baseline LSTM (default):

```
python app.py
```

Topic-conditioned LSTM (PowerShell):

```
$env:LSTM_MODEL       = "models\lstm_model_topic.pt"
$env:VOCAB_PATH       = "models\vocab_min5.pkl"
$env:LINE_COUNTS_PATH = "models\line_counts_min5.pkl"
$env:TOPIC_STATS_PATH = "models\topic_stats_min5.pkl"
python app.py
```

### CLI

```
python main.py --keyword "পাহাড়" --lines 4 --temp 0.85
```

## Project structure

```
src/
  preprocessing.py      Bengali text cleaning
  vocabulary.py         Word to ID mapping
  word2vec_model.py     Skip-gram embeddings
  rhyme_controller.py   AABB rhyme planning
  lstm_model.py         LSTM architecture
  lstm_generator.py     Poem generation engine
  poetic_enhancer.py    Grammar heuristics
  trigram_model.py      Traditional baseline
  generator.py          Trigram generation
templates/              HTML
static/                 CSS and JS
models/                 Trained models (not in repo)
data/                   Corpus (not in repo)
app.py                  Flask web interface
main.py                 CLI entry point
train_lstm_only.py      Training script
check_topic.py          Quick test utility
```

## Results

| Model | PPL | TTR | Rhyme Acc | Theme Sim | Grammar |
|---|---|---|---|---|---|
| Trigram LM | 84.6 | 0.612 | 31.2% | 0.412 | 58.4% |
| BanglaLSTM (Greedy) | 51.2 | 0.540 | 24.8% | 0.485 | 71.2% |
| Full Guided System | 42.8 | 0.742 | 91.4% | 0.884 | 95.8% |


## License

Educational use (KUET CSE 4122 NLP Lab Project).
By Md.Ashraful Hasan
CSE,KUET
