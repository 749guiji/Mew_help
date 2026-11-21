"""主题分类器模型:RoBERTa-wwm-ext(BERT 架构)裸编码器 + 手接 17 维 sigmoid 头。

注意:chinese-roberta-wwm-ext 实际是 BERT 架构,用 BertModel 加载(非 RobertaModel)。
"""

from __future__ import annotations

import torch.nn as nn
from transformers import BertModel


class TopicClassifier(nn.Module):
    def __init__(self, model_name_or_path: str, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name_or_path)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        out = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        pooled = out.last_hidden_state[:, 0]  # [CLS]
        return self.classifier(self.dropout(pooled))  # logits [B, num_labels]
