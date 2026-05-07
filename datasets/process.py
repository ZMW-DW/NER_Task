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
You are a high-precision semantic-role labeling (SRL) engine for Multi-Intent Task Dispatching.

Your goal is to parse unstructured user commands into structured Task Frames.
Each Task Frame is anchored by a Predicate (PRED). 
RULE: 1 Predicate Span = 1 Intent/Function Call.

---
## 0. STRICT ALIGNMENT CONSTRAINT (CRITICAL)

1. The number of labels MUST exactly equal the number of tokens.
2. Each token must be assigned exactly one BIO tag.
3. If alignment is not possible, the sample is INVALID and must be rejected.
4. Never output truncated or padded label sequences.

## 1. Tag Set (The SLU Schema)

- B-PRED / I-PRED : Actions, including phrasal verbs & auxiliaries (e.g., "turn off", "switch to", "want to open").
- B-OBJ / I-OBJ   : Primary targets or entities (e.g., "light", "HDMI 1", "whiteboard", "volume").
- B-VAL / I-VAL   : Values, states, or specific modes (e.g., "50%", "active", "silent", "blue").
- B-LOC / I-LOC   : Locations, origins, or destinations (e.g., "living room", "on the screen").
- B-ATTR / I-ATTR : Modifiers or specific properties (e.g., "this", "first", "large", "my").
- O               : Connectors, fillers, or irrelevant tokens.

---

## 2. Core Operational Principles

### 2.1 The Multi-Intent Split
If a sentence contains multiple verbs (Predicates), they MUST be labeled as distinct B-PRED spans.
Example: "Close the window and open the door" 
-> "Close"(B-PRED), "open"(B-PRED).

### 2.2 Predicate Continuity
Phrasal verbs and auxiliary chains MUST be a continuous PRED span.
- "Log in to" -> B-PRED, I-PRED, I-PRED
- "Need to switch off" -> B-PRED, I-PRED, I-PRED, I-PRED

### 2.3 Semantic Anchoring
Arguments (OBJ, VAL, LOC) belong to the nearest PRED in the syntactic flow. 

---

## 3. Formatting Rules

1. Use ONLY the provided tokens.
2. Every token must have exactly one BIO tag.
3. Return ONLY a valid JSON object.

---

## 4. Multi-Intent Example

### Input
Sentence: 
switch the living room light to blue and set brightness to 80%

Tokens:
["switch", "the", "living", "room", "light", "to", "blue", "and", "set", "brightness", "to", "80", "%"]

### Output
{{
  "tokens": ["switch", "the", "living", "room", "light", "to", "blue", "and", "set", "brightness", "to", "80", "%"],
  "labels": [
    "B-PRED", "O", "B-LOC", "I-LOC", "B-OBJ", "O", "B-VAL", "O", "B-PRED", "B-OBJ", "O", "B-VAL", "I-VAL"
  ]
}}

---

## 5. Your Task

Perform BIO tagging for the following input to facilitate multi-function dispatching:

Sentence:
{sentence}

Tokens:
{tokens}

Return ONLY the JSON output.
"""


def load_datasets(dataset: Path) -> list[dict]:
    data = []
    with dataset.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            item["tokens"] = item["input"].lower().split()
            data.append(item)
    return data

# def merge_data(datasets: list[dict], random_factor: int = 3) -> list[dict]:
#     merged = []
#     i = 0
#     n = len(datasets)

#     while i < n:
#         k = random.randint(2, random_factor)
#         batch = datasets[i:i + k]
#         if not batch:
#             break
#         merged_input = " ".join(x["input"] for x in batch)

#         merged_output = [x["output"] for x in batch]

#         merged_tokens = []
#         for x in batch:
#             merged_tokens.extend(x["tokens"])

#         merged.append({
#             "input": merged_input,
#             "output": merged_output,
#             "tokens": merged_tokens
#         })

#         i += k

#     return merged

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
        base_prompt = PROMPT.format_map({
            "sentence": dataset['input'],
            "tokens": dataset['tokens']
        })

        error_msg = ""
        for _ in range(2):
            messages = [{
                "role": "user",
                "content": base_prompt + error_msg
            }]

            response = await client.chat.completions.create(
                model=os.environ['MODEL'],
                messages=messages,
                response_format={'type': 'json_object'},
                extra_body={"thinking": {"type": "disabled"}}
            )

            try:
                labels = json.loads(response.choices[0].message.content)['labels']
            except Exception:
                error_msg = "\nPrevious output was invalid JSON. Regenerate strictly."
                continue

            if len(labels) == len(dataset['tokens']):
                dataset['labels'] = labels
                return dataset

            error_msg = (
                f"\nPrevious output was invalid."
                f"\nReason: label length {len(labels)} != token length {len(dataset['tokens'])}."
                f"\nYou MUST output exactly {len(dataset['tokens'])} labels."
            )

        dataset['labels'] = None
        dataset['error'] = "alignment_failed"
        return dataset



def build_train(debug_mode: bool = False):
    datasets = DATASETS / "datasets.jsonl"
    output = DATASETS / "train.json"
    clinet = AsyncClient(base_url=os.environ['BASE_URL'], api_key=os.environ['API_KEY'])
    tokenizer = BertTokenizer.from_pretrained(str(MODEL_PATH), fix_mistral_regex=True)
    # first_predata = load_datasets(tokenizer, datasets)
    first_predata = load_datasets(datasets)
    # seconde_predata = merge_data(first_predata)
    seconde_predata = first_predata
    final_predata = asyncio.run(async_llm(clinet, seconde_predata[:2] if debug_mode else seconde_predata))
    output.write_text(json.dumps(final_predata, indent=4, ensure_ascii=False), encoding="utf-8")

def build_test():
    datasets = DATASETS / "evaluate.jsonl"
    output = DATASETS / "test.json"
    clinet = AsyncClient(base_url=os.environ['BASE_URL'], api_key=os.environ['API_KEY'])
    tokenizer = BertTokenizer.from_pretrained(str(MODEL_PATH), fix_mistral_regex=True)
    # first_predata = load_datasets(tokenizer, datasets)
    first_predata = load_datasets(datasets)
    # seconde_predata = merge_data(first_predata)
    seconde_predata = first_predata
    final_predata = asyncio.run(async_llm(clinet, seconde_predata))
    output.write_text(json.dumps(final_predata, indent=4, ensure_ascii=False), encoding="utf-8")
    

if __name__ == "__main__":
    debug_mode = False
    build_train(debug_mode)
    # build_test()