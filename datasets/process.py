import os
import json
import random
import asyncio
from pathlib import Path
from tqdm import tqdm
from openai import AsyncClient
from asyncio import Semaphore
from transformers import BertTokenizer
from dotenv import load_dotenv
load_dotenv()

WORK_SPACE = Path(__file__).parent.parent
MODEL_PATH = WORK_SPACE / "bert-base-multilingual-uncased"
DATASETS = WORK_SPACE / "datasets"

semaphore = Semaphore(5)

PROMPT = """
You are a high-precision syntactic-semantic sequence labeling system.

This is NOT traditional NER.

Your task is to assign BIO tags to tokens based on:
- predicate structure
- argument structure
- phrasal verb semantics
- multi-intent decomposition

---

## 1. Input

You are given:
1. A sentence
2. Its tokenized form (WordPiece / BERT tokens)

You MUST strictly use the given tokens.

---

## 2. Core Principle

The sentence is interpreted as a set of **semantic predicate frames**.

Each predicate frame contains:
- Predicate (action / verb phrase)
- Subject (who performs action)
- Object (target / destination / entity)
- Attributes (modifiers)

A sentence may contain MULTIPLE predicate frames.

---

## 3. CRITICAL: Predicate Definition (VERY IMPORTANT)

### 3.1 Predicate is NOT a single verb

A predicate may be:

- single verb:
  "open"
  "close"

- phrasal verb (VERY IMPORTANT):
  "get back to"
  "log in to"
  "switch on"
  "turn off"
  "move into"
  "go back to"

- auxiliary + verb structure:
  "want to exit"
  "need to close"
  "have to switch"

### 3.2 Rule

👉 ALL tokens belonging to a predicate MUST be labeled as PRED (continuous span)

👉 DO NOT split verbs and particles inside predicate phrases

---

## 4. Tag Set (ONLY these labels allowed)

- B-SUB / I-SUB → Subject
- B-PRED / I-PRED → Predicate (INCLUDING phrasal verbs + auxiliaries)
- B-OBJ / I-OBJ → Object / target / destination
- B-ATT / I-ATT → Attribute / modifier
- O → Outside any semantic role

---

## 5. BIO Rules

- B-XXX = first token of a span
- I-XXX = inside span
- Each token must have exactly ONE label
- No skipping tokens
- No merging tokens
- Must strictly follow input token order

---

## 6. Multi-Intent Rule

If multiple actions exist:
- Each predicate defines one intent
- BUT output must remain a single BIO sequence

---

## 7. Important Linguistic Rules

### 7.1 "to" is NOT a separator
Examples:
- "get back to"
- "switch to"
- "log in to"

👉 "to" is part of predicate or argument depending on structure

---

### 7.2 Determiners are not objects
- "the", "a", "this", "that" → usually O or part of OBJ span

---

### 7.3 Object spans
Objects include:
- nouns
- noun phrases
- destinations
- device names
- UI elements (screen, board, session, HDMI1)

---

## 8. Output format (STRICT)

Return ONLY valid JSON:

{{
  "tokens": [...],
  "labels": [...]
}}

---

## 9. Example

### Input

Sentence:
i want to exit the board mode close this whiteboard session

Tokens:
["i", "want", "to", "exit", "the", "board", "mode", "close", "this", "whiteboard", "session"]

---

### Output

{{
  "tokens": ["i", "want", "to", "exit", "the", "board", "mode", "close", "this", "whiteboard", "session"],
  "labels": [
    "B-SUB",
    "B-PRED",
    "I-PRED",
    "I-PRED",
    "B-OBJ",
    "I-OBJ",
    "I-OBJ",
    "B-PRED",
    "B-OBJ",
    "I-OBJ",
    "I-OBJ"
  ]
}}

---

## 10. Your Task

Now perform BIO tagging for the following input:

Sentence:
{sentence}

Tokens:
{tokens}

Return ONLY the JSON output.
"""

def load_datasets(tokenizer: BertTokenizer, dataset: Path) -> list[dict]:
    DATASETS = []
    with dataset.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            DATASETS.append(json.loads(line))
    return split_to_token(tokenizer, DATASETS)

def split_to_token(
    tokenizer: BertTokenizer,
    datasets: list[dict],
    batch_size: int = 8
) -> list[dict]:

    for i in range(0, len(datasets), batch_size):
        batch = datasets[i:i + batch_size]
        texts = [x['input'] for x in batch]
        encoded = tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True
        )
        input_ids = encoded['input_ids']

        for idx, data in enumerate(batch):
            ids = input_ids[idx]
            tokens = tokenizer.convert_ids_to_tokens(ids)
            tokens = [t for t in tokens if t not in ["[PAD]", "[SEP]", "[CLS]"]]
            data['tokens'] = tokens

    return datasets

def merge_data(DATASETS: list[dict], random_factor: int = 3) -> list[dict]:
    merged = []
    i = 0
    n = len(DATASETS)

    while i < n:
        k = random.randint(2, random_factor)
        batch = DATASETS[i:i + k]
        if not batch:
            break
        merged_input = " ".join(x["input"] for x in batch)

        merged_output = [x["output"] for x in batch]

        merged_tokens = []
        for x in batch:
            merged_tokens.extend(x["tokens"])

        merged.append({
            "input": merged_input,
            "output": merged_output,
            "tokens": merged_tokens
        })

        i += k

    return merged

async def async_llm(client: AsyncClient, datasets: list[dict]):
    tasks = [
        async_process(
            client, 
            dataset
        ) 
        for dataset in datasets
    ]

    results = []
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        result = await coro
        results.append(result)
    assert len(datasets) == len(results)

    return results
    
async def async_process(client: AsyncClient, dataset: dict) -> dict:
    async with semaphore:
        messages = [
            {
                "role": "user", 
                "content": PROMPT.format_map(
                    {
                        "sentence" : dataset['input'],
                        "tokens": dataset['tokens']
                    }
                )
            }
        ]
        response = await client.chat.completions.create(
            model=os.environ['MODEL'],
            messages=messages,
                    response_format={
                'type': 'json_object'
            },
            extra_body={"thinking": {"type": "disabled"}}
        )

        dataset['labels'] = json.loads(response.choices[0].message.content)['labels']
        return dataset


def build_train(debug_mode: bool = False):
    datasets = DATASETS / "datasets.jsonl"
    output = DATASETS / "tran.json"
    clinet = AsyncClient(base_url=os.environ['BASE_URL'], api_key=os.environ['API_KEY'])
    tokenizer = BertTokenizer.from_pretrained(str(MODEL_PATH), fix_mistral_regex=True)
    first_predata = load_datasets(tokenizer, datasets)
    seconde_predata = merge_data(first_predata)
    final_predata = asyncio.run(async_llm(clinet, seconde_predata[:2] if debug_mode else seconde_predata))
    output.write_text(json.dumps(final_predata, indent=4, ensure_ascii=False), encoding="utf-8")

def build_test():
    datasets = DATASETS / "evaluate.jsonl"
    output = DATASETS / "test.json"
    clinet = AsyncClient(base_url=os.environ['BASE_URL'], api_key=os.environ['API_KEY'])
    tokenizer = BertTokenizer.from_pretrained(str(MODEL_PATH), fix_mistral_regex=True)
    first_predata = load_datasets(tokenizer, datasets)
    seconde_predata = merge_data(first_predata)
    final_predata = asyncio.run(async_llm(clinet, seconde_predata))
    output.write_text(json.dumps(final_predata, indent=4, ensure_ascii=False), encoding="utf-8")
    

if __name__ == "__main__":
    debug_mode = False
    build_train(debug_mode)
    build_test()