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
    
    def _load_wav_list(list_path, wav_root):
        with open(list_path, "r") as f:
            return [os.path.join(wav_root, line.strip().lstrip("/")) for line in f if line.strip()]
    
    def _calc_norm_stats(dataset, n_stats: int = 100_000, label: str = "train"):
        n_stats = min(n_stats, len(dataset))
        idx = np.random.choice(len(dataset), size=n_stats, replace=False)
        feats = [dataset[i] for i in tqdm(idx, desc=f"[{label} stats]")]
        flat  = np.hstack(feats)
        mean, std = flat.mean(), flat.std()
        print(f"[{label}] mean={mean:.4f}, std={std:.4f}")
        return [mean, std]

    list_path = cfg.get(split_key)
    split_label = split_key.replace("_list", "")
    wav_list = _load_wav_list(list_path, cfg["wav_dir"])
    ds_cfg = SimpleNamespace(**cfg["dataset"])
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

