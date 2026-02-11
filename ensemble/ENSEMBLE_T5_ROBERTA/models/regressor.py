import torch.nn as nn
from transformers import AutoModel

class EncoderRegressor(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.enc = AutoModel.from_pretrained(model_name)
        self.drop = nn.Dropout(0.1)
        self.head = nn.Linear(self.enc.config.hidden_size, 1)

    def forward(self, ids, mask):
        x = self.enc(ids, mask).last_hidden_state[:, 0]
        x = self.drop(x)
        return self.head(x).squeeze(-1)
