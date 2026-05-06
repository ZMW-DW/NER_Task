import torch
import torch.nn as nn
import torch.functional as f
from dataclasses import dataclass
from pathlib import Path
from transformers import BertTokenizer, BertModel

@dataclass
class BaseLineModelConfig:
    entropy_numbers: int = 



class BaseLineModelOutput:
    logits: torch.Tensor

class BaseLineModel(nn.Module):
    def __init__(self, arg: BaseLineModelConfig):
        super().__init__()
        self.arg = arg

    def _setup(self):
        self.tokenizer = BertTokenizer.from_pretrained()
        self.backbone = BertModel.from_pretrained()