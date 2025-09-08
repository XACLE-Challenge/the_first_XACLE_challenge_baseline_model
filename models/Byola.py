import torch
import torch.nn as nn
from tqdm import tqdm
import nnAudio.features
import numpy as np
import os
from types import SimpleNamespace

from .byola.v2.byol_a2.models import AudioNTT2022, load_pretrained_weights
from .byola.v2.byol_a2.augmentations import PrecomputedNorm
from .byola.byol_a.dataset import WaveInLMSOutDataset

def get_normalizer(cfg, split_key):
    
    def _load_wav_list(list_path, wav_dir):
        import csv
        wav_list = []
        with open(list_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                if row and row[0].strip():
                    wav_list.append(os.path.join(wav_dir, row[0].strip()))
        return wav_list
    
    def _calc_norm_stats(dataset, n_stats: int = 100_000, label: str = "train"):
        n_stats = min(n_stats, len(dataset))
        idx = np.random.choice(len(dataset), size=n_stats, replace=False)
        feats = [dataset[i] for i in tqdm(idx, desc=f"[{label} stats]")]
        flat  = np.hstack(feats)
        mean, std = flat.mean(), flat.std()
        print(f"[{label}] mean={mean:.4f}, std={std:.4f}")
        return [mean, std]

    list_path = cfg.get(split_key)                      # metadata list 
    split_label = split_key.replace("_list", "")        # train or validation
    wav_list = _load_wav_list(list_path, os.path.join(cfg["wav_dir"], split_label))
    ds_cfg = SimpleNamespace(**cfg["audio_encoder"]["dataset"])
    stats_ds = WaveInLMSOutDataset(ds_cfg, audio_files=wav_list, labels=None, tfms=None)
    stats = _calc_norm_stats(stats_ds, label=split_label)
    return PrecomputedNorm(stats)

class Byola(nn.Module):
    def __init__(self, model_path, device, n_mels, feature_d, sr, n_fft, win_length, hop_length, fmin, fmax):
        super().__init__()
        self.device = device
        self.model_path = model_path
        self.n_mels = n_mels
        self.feature_d = feature_d
        self.sr= sr
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax

        self.model = AudioNTT2022(n_mels=self.n_mels, d=self.feature_d).to(self.device)
        load_pretrained_weights(self.model, self.model_path)
        self.model.eval()
        self.to_melspec = nnAudio.features.MelSpectrogram(
            sr = self.sr,
            n_fft =  self.n_fft,
            win_length = self.win_length,
            hop_length = self.hop_length,
            n_mels = self.n_mels,
            fmin = self.fmin,
            fmax = self.fmax,
            center = True,
            power  = 2,
            verbose = False
        )

    def forward(self, wavs, normalizer):
        lms = normalizer((self.to_melspec(wavs) + torch.finfo(torch.float).eps).log()).to(self.device)  # (B, n_mels, T)
        lms = lms.unsqueeze(1)
        feat = self.model(lms)
        return feat

