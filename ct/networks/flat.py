import torch
import torch.nn as nn
import torch.nn.functional as F


class FMLP(nn.Module):
    def __init__(self, d_in, c_in, dropout, sigm=True):
        super(FMLP, self).__init__()
        self.d_in = d_in
        self.c_in = c_in
        self.sigm = sigm
        self.seq1 = nn.Sequential(
            nn.Linear(self.d_in, 5),
            nn.Linear(5, 10),
            nn.LeakyReLU(True),
            nn.Dropout(dropout),
            nn.BatchNorm1d(10),
        )
        self.seq2 = nn.Sequential(
            nn.Linear(self.c_in * 10, 20),
            nn.LeakyReLU(True),
            nn.Dropout(dropout),
            nn.BatchNorm1d(20),
            nn.Linear(20, 1),
        )

    def forward(self, x):
        b, c, d = x.shape
        assert c == self.c_in
        assert d == self.d_in
        device = x.device
        out = torch.empty(b, 10*c).to(device)
        for i in range(c):
            _x = x[:, i, :]
            _x = self.seq1(_x)
            out[:, i*10: (i+1)*10] = _x

        out = self.seq2(out)
        if self.sigm:
            out = torch.sigmoid(out)

        return out.view(-1)


