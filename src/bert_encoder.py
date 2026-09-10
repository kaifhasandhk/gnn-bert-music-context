import torch
import torch.nn as nn
from transformers import DistilBertModel

class BertTagClassifier(nn.Module):
    """Task 1: DistilBERT Tag-Context Classifier."""
    def __init__(self, num_tags: int = 50, model_name: str = 'distilbert-base-uncased'):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(model_name)
        self.head = nn.Linear(self.bert.config.hidden_size, num_tags)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = out.last_hidden_state[:, 0]
        logits = self.head(cls_token)
        return logits
