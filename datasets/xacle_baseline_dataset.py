import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import torchaudio

def get_dataset(txt_file_path, wav_dir, tokenizer, max_sec, sr, org_max=10.0, org_min=0.0):
    return XACLEDataset(txt_file_path, wav_dir, tokenizer, max_sec, sr, org_max, org_min)

def get_infdataset(txt_file_path, wav_dir, tokenizer, max_sec, sr):
    return XACLEINFDataset(txt_file_path, wav_dir, tokenizer, max_sec, sr)

from torch.nn.functional import pad as pad1d 

class XACLEDataset(Dataset):
    def __init__(
            self,
            txt_file_path: str,
            wav_dir : str,
            tokenizer,
            max_sec: int = 10,
            sr: int = 16_000,
            org_max: float = 10.0,
            org_min: float = 0.0
    ):
        super().__init__()
        df = pd.read_csv(txt_file_path)
        self.wav_dir = wav_dir
        self.tokenizer = tokenizer
        self.wav_max_len = int(max_sec * sr)
        self.org_mid = (org_max + org_min) / 2
        self.norm_denom = 5.0
        bins = [-1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        labels = [f"{i}-{i+1}" for i in range(0, 10)]
        df["rating_category"] = pd.cut(df["average_score"], bins=bins, labels=labels)
        df["num_class"] = df.groupby("rating_category")["rating_category"].transform("count")
        self.df = df
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        wav_path        = os.path.join(self.wav_dir, r["wav_file_name"].lstrip("/"))
        wav, _          = torchaudio.load(wav_path)
        mos             = float(r["average_score"])
        mos_norm        = (mos - self.org_mid) / self.norm_denom
        caption         = r["text"]
        num_class       = int(r["num_class"])

        return dict(
            wav         = wav,
            score       = mos_norm,
            caption     = caption,
            num_class   = num_class,
            wav_path    = wav_path
        )
    
    def collate_fn(self, batch):
        # --- wav padding / trimming --------------------------------------
        wav_max_len = self.wav_max_len
        wavs        = [b["wav"] for b in batch]
        wav_fixed   = []
        for wav in wavs:
            pad_len = wav_max_len - wav.shape[1]
            if pad_len > 0:
                padded = pad1d(wav, (0, pad_len), mode="constant", value=0.0)
            else:
                padded = wav[:, :wav_max_len]
            wav_fixed.append(padded)
        wav_batch = torch.stack(wav_fixed)

        # --- simple tensors ----------------------------------------------
        mos_score = torch.tensor([b["score"]        for b in batch], dtype=torch.float)
        num_class = torch.tensor([b["num_class"]    for b in batch], dtype=torch.long)

        # --- caption tokens ----------------------------------------------
        captions   = [b["caption"] for b in batch]
        cap_tokens = self.tokenizer(captions, padding=True, return_tensors="pt")

        return dict(
            wavs            = wav_batch,
            scores          = mos_score,
            caption_tokens  = cap_tokens,
            num_class       = num_class,
            wav_paths       = [b["wav_path"] for b in batch]
        )
    
class XACLEINFDataset(Dataset):
    def __init__(
            self,
            txt_file_path: str,
            wav_dir : str,
            tokenizer,
            max_sec: int = 10,
            sr: int = 16_000
    ):
        super().__init__()
        df = pd.read_csv(txt_file_path)
        self.wav_dir = wav_dir
        self.tokenizer = tokenizer
        self.wav_max_len = int(max_sec * sr)
        self.df = df
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        wav_path = os.path.join(self.wav_dir, r["wav_file_name"].lstrip("/"))
        wav, _   = torchaudio.load(wav_path)
        caption  = r["text"]
        return dict(
            wav      = wav,
            caption  = caption,
            wav_path = wav_path
        )
    
    def collate_fn(self, batch):
        # --- wav padding / trimming --------------------------------------
        wav_max_len = self.wav_max_len
        wavs        = [b["wav"] for b in batch]
        wav_fixed   = []
        for wav in wavs:
            pad_len = wav_max_len - wav.shape[1]
            if pad_len > 0:
                padded = pad1d(wav, (0, pad_len), mode="constant", value=0.0)
            else:
                padded = wav[:, :wav_max_len]
            wav_fixed.append(padded)
        wav_batch = torch.stack(wav_fixed)

        # --- caption tokens ----------------------------------------------
        captions   = [b["caption"] for b in batch]
        cap_tokens = self.tokenizer(captions, padding=True, return_tensors="pt")

        return dict(
            wavs            = wav_batch,
            caption_tokens  = cap_tokens,
            wav_paths       = [b["wav_path"] for b in batch]
        )