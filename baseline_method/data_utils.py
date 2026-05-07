import torch
import json
from pathlib import Path
from transformers import BertTokenizer
from torch.utils.data import Dataset
from .comment import entropy2id

class BaseLineDataset(Dataset):
    def __init__(self, data_path: str | Path):
        super().__init__()
        if isinstance(data_path, str):
            data_path: Path = Path(data_path)
            if not data_path.exists():
                raise ValueError(f"Path error: {data_path}")
        self.data = self._load_and_format(data_path)
    
    def _load_and_format(self, data_path: Path):
        with data_path.open(encoding="utf-8") as f:
            raw_data = json.load(f)
        return [
            {
                "text": item['input'],
                "tokens": item["tokens"],
                "labels": item['labels']
            }
            for item in raw_data
        ]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
    
def collate_fn_factory(tokenizer: BertTokenizer, max_len: int = 256):

    def collate_fn(batch):
        input_ids_list = []
        attention_mask_list = []
        labels_list = []

        for item in batch:

            encoding = tokenizer(
                item["tokens"],
                is_split_into_words=True,   
                padding="max_length",
                truncation=True,
                max_length=max_len,
                return_tensors="pt"
            )

            input_ids = encoding["input_ids"].squeeze(0)
            attention_mask = encoding["attention_mask"].squeeze(0)

            word_ids = encoding.word_ids()

            # ========= label 对齐 =========
            aligned_labels = []
            prev_word = None

            for w in word_ids:
                if w is None:
                    aligned_labels.append(-100)  # CLS / SEP / PAD
                elif w != prev_word:
                    aligned_labels.append(entropy2id[item["labels"][w]])
                else:
                    aligned_labels.append(-100)  # subword
                prev_word = w

            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            labels_list.append(torch.tensor(aligned_labels, dtype=torch.long))

        return {
            "input_ids": torch.stack(input_ids_list),
            "attention_mask": torch.stack(attention_mask_list),
            "labels": torch.stack(labels_list)
        }

    return collate_fn

__all__ = [
    "BaseLineDataset",
    "collate_fn_factory"
]


