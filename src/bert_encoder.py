import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class TextContextBERT(nn.Module):
    """Task 1: DistilBERT Encoder for semantic natural language tag context."""
    def __init__(self, model_name: str = "distilbert-base-uncased", num_tags: int = 50):
        super(TextContextBERT, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(768, num_tags)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Extract full sequence representations [B, L, 768]
        h_text = outputs.last_hidden_state 
        # Extract [CLS] token pooling vector [B, 768]
        cls_token = h_text[:, 0, :] 
        probs = torch.sigmoid(self.classifier(cls_token))
        return h_text, cls_token, probs
