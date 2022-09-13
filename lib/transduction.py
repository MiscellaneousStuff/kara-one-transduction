import torch.nn as nn
import torch.nn.functional as F

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