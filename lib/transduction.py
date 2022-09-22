import torch.nn as nn
import torch.nn.functional as F

from .transformer import *

class DigitalVoicingModel(nn.Module):
    def __init__(self,
                 ins,
                 model_size,
                 n_layers,
                 dropout,
                 outs):
        super().__init__()
        self.dropout = dropout
        self.lstm = \
            nn.LSTM(
                ins, model_size, batch_first=True,
                bidirectional=True, num_layers=n_layers,
                dropout=dropout)
        self.w1 = nn.Linear(model_size * 2, outs)

    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x, _ = self.lstm(x)
        x = F.dropout(x, self.dropout, training=self.training)
        return self.w1(x)


class ProposedModel(nn.Module):
    def __init__(self,
                 model_size,
                 dropout=0.2,
                 num_layers=6,
                 n_heads=8,
                 dim_feedforward=3072,
                 out_dim=80):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(
            d_model=model_size,
            nhead=n_heads,
            relative_positional=True,
            relative_positional_distance=100,
            dim_feedforward=dim_feedforward,
            dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.w_out = nn.Linear(model_size, out_dim)
    
    def forward(self, x):
        x = x.transpose(0, 1)
        x = self.transformer(x)
        x = x.transpose(0, 1)
        x = self.w_out(x)
        return x