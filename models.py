# models.py
import torch
import torch.nn as nn
from transformers import BertModel

class StanceBERT(nn.Module):
    def __init__(self, num_aspects: int, model_name: str = "bert-base-uncased", hidden: int = 256):
        super().__init__()

        self.bert = BertModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size

        self.aspect_proj = nn.Sequential(
            nn.Linear(num_aspects, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        self.head = nn.Sequential(
            nn.Linear(hidden_size + hidden, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_aspects),
        )

    def forward(self, input_ids, attention_mask, token_type_ids, word_one_hot):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls = out.last_hidden_state[:, 0, :]
        asp = self.aspect_proj(word_one_hot)
        fused = torch.cat([cls, asp], dim=-1)
        logits = self.head(fused)
        return logits
