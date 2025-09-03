import torch
import torch.nn as nn
from .Roberta import RoBERTa as TextEncoder
from .Byola import Byola as AudioEncoder

class XACLEBenchmarkModel(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        self.device = device
        ## Text Encoder
        self.text_encoder = TextEncoder(
            model_name  = cfg["roberta"]["pretrained_model"],
            device      = self.device
        )
        ## Audio Encoder
        self.audio_encoder = AudioEncoder(
            model_path    = cfg["byola"]["byola_model"],
            device        = self.device,
            n_mels        = cfg["byola"]["n_mels"],
            feature_d     = cfg["byola"]["feature_d"],
            sr            = cfg["byola"]["sample_rate"],
            n_fft         = cfg["byola"]["n_fft"],
            win_length    = cfg["byola"]["win_length"],
            hop_length    = cfg["byola"]["hop_length"],
            fmin          = cfg["byola"]["f_min"],
            fmax          = cfg["byola"]["f_max"]
        )
        ## LDConditioner
        self.ldconditioner = LDConditioner(
            input_dim        = cfg['model']['conditioner']['input_dim'],
            rnn_hidden_size  = cfg['model']['conditioner']['rnn_hidden_size'],
            rnn_num_layers   = cfg['model']['conditioner']['rnn_num_layers']
        )
        ## Projection
        self.projection    = Projection(
            input_dim       = cfg['model']['projection']['input_dim'],
            hidden_dim      = cfg['model']['projection']['hidden_dim'],
            activation      = cfg['model']['projection']['activation'],
            range_clipping  = cfg['model']['projection']['range_clipping'],
            dropout         = cfg['model']['projection']['dropout']
        )

    def forward(self, batch: dict, normalizer):
        # text embeddig (B, 1024)
        txt_emb   = self.text_encoder(batch["caption_tokens"])
        # audio embedding (B, 250, 3072)
        audio_emb = self.audio_encoder(batch["wavs"], normalizer)
        # concat feature
        cond_emb  = self.ldconditioner(audio_emb, txt_emb)
        # score predict
        mos_hat   = self.projection(cond_emb)

        return mos_hat

class LDConditioner(nn.Module):
    def __init__(self, input_dim, rnn_hidden_size, rnn_num_layers,
                 batch_first=True, bidirectional=True):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size    = input_dim,              # 4096
            hidden_size   = rnn_hidden_size,        #  512
            num_layers    = rnn_num_layers,         #    1
            batch_first   = batch_first,            # True
            bidirectional = bidirectional           # True
        )
        self.out_dim = rnn_hidden_size * (2 if bidirectional else 1)    # 1024

    def forward(self, audio_emb, text_emb):
        """
        audio_emb       : (B, T, D_a)  --  Byola feature per frame
        text_emb        : (B, D_t)     --  RoBERTa CLS embedding
        """
        txt_expand = text_emb.unsqueeze(1).expand(-1, audio_emb.size(1), -1)
        feat = torch.cat([audio_emb, txt_expand], dim=2)                # (B, T, D_a+D_t)

        # Bi-LSTM
        out, _ = self.rnn(feat)     # (B, T, 1024)
        first  = out[:, 0, :]       # (B, 1024)
        last   = out[:, -1, :]      # (B, 1024)
        return (first + last) / 2   # (B, 1024)
    
class Projection(nn.Module):
    def __init__(self, input_dim, hidden_dim,
                 activation: str = "ReLU",
                 range_clipping: bool = False, dropout: float = .3):
        super().__init__()
        self.range_clipping = range_clipping
        self.act = getattr(nn, activation)() if isinstance(activation, str) else activation 
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            self.act,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        if self.range_clipping:
            self.out_act = nn.Tanh()

    def forward(self, x):
        """
        x : (B, 1024)
        """
        x = self.net(x) # (B, 1)
        if self.range_clipping:
            x = self.out_act(x)
        x = x.squeeze(-1)   # (B, )
        return x