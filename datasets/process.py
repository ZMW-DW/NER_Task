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
You are a high-precision Named Entity Recognition (NER) annotation system.

## Task
You are given:
1. A sentence
2. Its tokenized form (WordPiece/BERT tokens)

Your task is to perform BIO tagging based on syntactic-semantic roles.

## Important Rules

### 1. Multi-intent structure
A sentence may contain multiple predicates (verbs/actions).  
Each predicate corresponds to an independent intent.

You MUST:
- Identify ALL predicates (verbs/actions)
- Treat each predicate and its arguments as a separate semantic unit
- Still produce a single unified BIO tag sequence

---

### 2. Tag set (ONLY these labels are allowed)

- B-SUB / I-SUB → Subject (主语)
- B-PRED / I-PRED → Predicate / Verb (谓语 / 动作)
- B-OBJ / I-OBJ → Object (宾语)
- B-ATT / I-ATT → Attribute / Modifier (定语)

---

### 3. BIO Rules
- B-XXX = beginning of a span
- I-XXX = inside a span
- Each token must have exactly ONE label
- Labels must align strictly with tokens (no merging, no skipping)

---

### 4. Tokenization constraint
You MUST use the given token list exactly.
Do NOT modify, merge, or re-tokenize.

---

### 5. Output format (STRICT)

Return JSON only:

{{
  "tokens": [...],
  "labels": [...]
}}

---

## Example

### Input

Sentence:
"switch to hdmi one change input to type c port 2"

Tokens:
['switch', 'to', 'hd', '##mi', 'one', 'change', 'input', 'to', 'type', 'c', 'port', '2']

---

### Output

{{
  "tokens": ["switch", "to", "hd", "##mi", "one", "change", "input", "to", "type", "c", "port", "2"],
  "labels": [
    "B-PRED", "O", "B-OBJ", "I-OBJ", "I-OBJ",
    "B-PRED", "B-OBJ", "O", "B-OBJ", "I-OBJ", "I-OBJ", "I-OBJ"
  ]
}}

---

## Your Task

Now perform NER tagging for the following input:

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
    DATASETS: list[dict],
    batch_size: int = 8
) -> list[dict]:

    for i in range(0, len(DATASETS), batch_size):
        batch = DATASETS[i:i + batch_size]
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

    return DATASETS

def merge_data(DATASETS: list[dict], random_factor: int = 3) -> list[dict]:
    merged = []
    i = 0
    n = len(DATASETS)

    while i < n:
        k = random.randint(1, random_factor)
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
    # build_test()