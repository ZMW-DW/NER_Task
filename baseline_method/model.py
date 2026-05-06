import torch
import torch.nn as nn
from dataclasses import dataclass
from pathlib import Path
from transformers import BertModel
from .comment import ENTROPY


# =========================
# Config 配置类
# =========================
@dataclass
class BaseLineModelConfig:
    """
    模型配置类，用于统一管理超参数与运行环境
    """

    is_train: bool = False  # 是否训练模式（影响dropout等行为）

    # 预训练BERT模型路径（本地加载）
    backbone_model: Path = Path(__file__).parent.parent / "bert-base-multilingual-uncased"

    hidden_size: int = 768  # BERT隐层维度

    @property
    def num_labels(self) -> int:
        "标签类别数(NER标签数)"
        return len(ENTROPY)     

    @property
    def dropout(self) -> float:
        """
        Dropout策略：
        - 训练阶段使用0.1防止过拟合
        - 推理阶段关闭dropout保证稳定性
        """
        return 0.1 if self.is_train else 0.0

    # 自动选择运行设备
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    


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
        labels: list[list[str]],
        predictions: list[list[str]]
    ):
        """
        参数说明：
        - logits: 模型原始输出（未经过softmax） [B, L, C]
        - labels: 原始标签（字符串形式）
        - predictions: 预测标签（字符串形式）
        """
        self.logits = logits
        self.labels = labels
        self.predictions = predictions



# =========================
# Baseline（BERT + Token Classification）
# =========================
class BaseLineModel(nn.Module):
    def __init__(self, config: BaseLineModelConfig):
        super().__init__()

        self.config = config
        self.device = config.device

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
            config.num_labels
        ).to(self.device)

        # Dropout层（防止过拟合）
        self.dropout = nn.Dropout(config.dropout)

    # =========================
    # 前向传播
    # =========================
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: list[list[str]],
    ) -> BaseLineModelOutput:
        """
        参数：
        - input_ids: token id序列 [B, L]
        - attention_mask: attention mask [B, L]
        - labels: 原始标签（用于返回，不参与forward计算）

        返回：
        - BaseLineModelOutput
        """

        # 移动到设备
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        # ===== BERT编码 =====
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # 取最后一层hidden state [B, L, H]
        hidden = outputs.last_hidden_state

        # dropout（训练时生效）
        hidden = self.dropout(hidden)

        # ===== Token级分类 =====
        logits = self.classifier(hidden)  # [B, L, C]

        # ===== 解码预测结果 =====
        pred_ids = torch.argmax(logits, dim=-1).cpu().tolist()

        # 将id映射为标签字符串
        predictions = [
            [ENTROPY[idx] for idx in seq]
            for seq in pred_ids
        ]

        return BaseLineModelOutput(
            logits=logits,
            labels=labels,
            predictions=predictions,
        )


    # =========================
    # 损失函数（NER标准写法）
    # =========================
    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor
    ):
        """
        参数：
        - logits: 模型输出 [B, L, C]
        - labels: 标签id [B, L]
        - attention_mask: mask [B, L]

        说明：
        - 使用 CrossEntropyLoss
        - 忽略 padding 和特殊token（label = -100）
        """

        # ignore_index=-100 是 HuggingFace 标准做法
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

        # ===== 仅计算有效token =====
        active_loss = attention_mask.view(-1) == 1

        # reshape
        active_logits = logits.view(-1, logits.shape[-1])
        active_labels = labels.view(-1)

        # 过滤padding位置
        active_logits = active_logits[active_loss]
        active_labels = active_labels[active_loss]

        # 计算损失
        return loss_fct(active_logits, active_labels)