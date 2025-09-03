import torch
import torch.nn as nn
from transformers import RobertaModel


class RoBERTa(nn.Module):
    def __init__(self, model_name, device):
        super().__init__()
        self.device = device
        self.model_name = model_name
        self.model = RobertaModel.from_pretrained(self.model_name, cache_dir="./hf_cache").to(self.device)

    def forward(self, input_tokens):
        outputs = self.model(**input_tokens)
        return outputs.last_hidden_state[:, 0, :]
