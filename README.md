# Multi-Task Schedule / NER Annotation Pipeline

This project implements a **multi-intent NER annotation system** based on BERT tokenization and LLM-based BIO labeling.  
It supports dataset preprocessing, token alignment, data merging, and asynchronous LLM annotation.

---

## 🚀 Features

- JSONL dataset loader
- BERT WordPiece token alignment
- Multi-intent sentence merging
- BIO sequence labeling (SUB / PRED / OBJ / ATT)
- Async LLM annotation pipeline
- tqdm progress tracking
- Reproducible environment via `uv`

---

## 📁 Project Structure

```

.
├── basemodel.py              # baseline model implementation (to be extended)
├── main.py                   # training / inference entry
├── datasets/
│   ├── datasets.jsonl       # training data
│   ├── evaluate.jsonl       # evaluation data
│   ├── process.py           # dataset processing pipeline
│   ├── tran.json            # processed training data
│   └── test.json            # processed test data
├── bert-base-multilingual-uncased/
│   ├── config.json
│   ├── tokenizer.json
│   └── tokenizer_config.json
├── setup.sh                  # environment bootstrap script
├── pyproject.toml           # uv project config
└── uv.lock

````

---

## ⚙️ Environment Setup

### 1. Make setup script executable

```bash
chmod +x setup.sh
````

---

### 2. Initialize environment & download model

```bash
./setup.sh
```

This script will:

* Create/restore Python environment using `uv`
* Install required dependencies
* Download BERT multilingual model via ModelScope
* Prepare tokenizer and model files

---

## 📦 Dataset Preparation

Raw dataset format (`datasets.jsonl`):

```json
{"input": "switch to hdmi one", "output": {...}}
{"input": "change input to type c port 2", "output": {...}}
```

After preprocessing:

* Tokenization via BERT WordPiece
* Multi-intent sentence merging
* Output saved to:

  * `tran.json`
  * `test.json`

---

## 🧠 Model Pipeline

### 1. Tokenization

Uses `BertTokenizer`:

* WordPiece alignment
* Removes `[CLS]`, `[SEP]`, `[PAD]`

---

### 2. Data Augmentation

Multiple commands are merged into a single training sample:

```
input: "switch to hdmi one change input to type c port 2"
```

---

### 3. LLM Annotation

Asynchronous API calls generate BIO labels:

* Subject (SUB)
* Predicate (PRED)
* Object (OBJ)
* Attribute (ATT)

---

## 🏗️ Baseline Model

The baseline model is implemented in:

```
basemodel.py
```

It will include:

* Simple sequence tagging model
* BERT encoder backbone
* BIO classification head
* Evaluation metrics (F1 / Accuracy)

---

## ▶️ Run Pipeline

### Build training data

```bash
python datasets/process.py
```

### Run full pipeline

```bash
python main.py
```

---

## 📊 Output Format

Final annotated sample:

```json
{
  "input": "switch to hdmi one",
  "tokens": ["switch", "to", "hd", "##mi", "one"],
  "labels": ["B-PRED", "O", "B-OBJ", "I-OBJ", "I-OBJ"]
}
```

---

## 🔧 Requirements

* Python ≥ 3.10
* uv
* torch
* transformers
* openai
* tqdm
* python-dotenv

---

## 📌 Notes

* Ensure API key is set in `.env`:

```bash
BASE_URL=...
API_KEY=...
MODEL=...
```

* Large batch LLM calls may require rate-limit tuning (`Semaphore`)

---

## 🚧 Future Work

* Replace LLM labeling with hybrid weak supervision
* Add CRF decoding layer
* Improve multi-intent decomposition
* Add evaluation script (F1 / seq accuracy)

---

## 📜 License

MIT License


