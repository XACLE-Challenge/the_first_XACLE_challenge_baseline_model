import json
import torch

def load_config(config_path = "config.json"):
    with open(config_path, 'r') as f:
        cfg = json.load(f)
    if cfg.get("device") == "cuda" and not torch.cuda.is_available():
        cfg["device"] = "cpu"
    return cfg