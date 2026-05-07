import torch
import torch.nn as nn
from dataclasses import dataclass
from pathlib import Path
from transformers import BertModel, BertTokenizer
from .comment import ENTROPY, id2entropy


# =========================
# Config 配置类
# =========================
@dataclass
class BaseLineModelConfig:
    """
    模型配置类，用于统一管理超参数与运行环境
    """

    is_train: bool = False  # 是否训练模式（影响dropout等行为）
    batch_size: int = 16
    epochs: int = 5
    lr: float = 5e-5
    @property
    def dropout(self) -> float:
        """
        Dropout策略：
        - 训练阶段使用0.1防止过拟合
        - 推理阶段关闭dropout保证稳定性
        """
        return 0.1 if self.is_train else 0.0

    # 预训练BERT模型路径（本地加载）
    backbone_model: Path = Path(__file__).parent.parent / "bert-base-multilingual-uncased"
    hidden_size: int = 768  # BERT隐层维度

    @property
    def num_labels(self) -> int:
        "标签类别数(NER标签数)"
        return len(ENTROPY)     

    # 自动选择运行设备
    @property
    def device(self) -> str:
        # 1. 优先检查 Apple Silicon MPS 加速
        if torch.backends.mps.is_available():
            return "mps"
        # 2. 其次检查 NVIDIA CUDA 加速
        elif torch.cuda.is_available():
            return "cuda"
        # 3. 最后退回到 CPU
        else:
            return "cpu"
        

# =========================
# 输出封装类
# =========================
class BaseLineModelOutput:
    """
    模型输出结构体，用于统一返回结果
    """
    def __init__(
        self, 
        logits: torch.Tensor,
        loss: torch.Tensor,
    ):
        """
        参数说明：
        - logits: 模型原始输出（未经过softmax） [B, L, C]
        - predictions: 预测标签（字符串形式）
        """
        self.logits = logits
        self.loss = loss



# =========================
# Baseline（BERT + Token Classification）
# =========================
class BaseLineModel(nn.Module):
    def __init__(self, config: BaseLineModelConfig):
        super().__init__()

        self.config = config
        self.device = config.device
        self.num_labels = config.num_labels

        # ===== BERT Backbone =====
        self.backbone = BertModel.from_pretrained(
            config.backbone_model
        ).to(self.device)

        # 冻结Embedding层（减少训练参数 & 防止破坏词向量空间）
        for param in self.backbone.embeddings.parameters():
            param.requires_grad = False

        # ===== 分类头（逐token分类）=====
        self.classifier = nn.Linear(
            config.hidden_size,
            self.num_labels
        ).to(self.device)

        # Dropout层（防止过拟合）
        self.dropout = nn.Dropout(config.dropout)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

    # =========================
    # 前向传播
    # =========================
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None
    ) -> BaseLineModelOutput:
        """
        参数：
        - input_ids: token id序列 [B, L]
        - attention_mask: attention mask [B, L]
        - labels: 原始标签（用于返回，不参与forward计算）

        返回：
        - BaseLineModelOutput
        """
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        if isinstance(labels, torch.Tensor):
            labels = labels.to(self.device)

        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        hidden = outputs.last_hidden_state
        hidden = self.dropout(hidden)
        logits = self.classifier(hidden)  # [B, L, C]

        active_loss = attention_mask.view(-1) == 1
        active_logits = logits.reshape(-1, self.num_labels)[active_loss]
        active_labels = labels.reshape(-1)[active_loss]

        loss = 0
        if isinstance(labels, torch.Tensor):
            loss = self.loss_fct(active_logits, active_labels)


        return BaseLineModelOutput(
            logits=logits,
            loss=loss,
        )
    
__all__ = [
    "BaseLineModelConfig",
    "BaseLineModel"
]