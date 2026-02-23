import torch
import torch.nn as nn
import pandas as pd
import os

class ECC:
    def __init__(self, repeats=3):
        self.repeats = repeats

    def encode(self, data):
        return data.repeat_interleave(self.repeats, dim=1)

    def decode(self, logits):
        B, T = logits.shape
        reshaped = logits.view(B, T // self.repeats, self.repeats)
        return (reshaped.sum(dim=2) > 0).float()

class ExcelLogger:
    def __init__(self, filename="log.xlsx"):
        self.filename = filename

    def log(self, epoch, metrics):
        row = {"Epoch": epoch, **metrics}
        df_new = pd.DataFrame([row])
        if not os.path.exists(self.filename):
            df_new.to_excel(self.filename, index=False)
        else:
            df = pd.read_excel(self.filename)
            pd.concat([df, df_new], ignore_index=True).to_excel(self.filename, index=False)

class HaarDWT(nn.Module):
    def forward(self, x):
        x01, x02 = x[:, :, 0::2, :] / 2, x[:, :, 1::2, :] / 2
        x1, x2 = x01[:, :, :, 0::2], x02[:, :, :, 0::2]
        x3, x4 = x01[:, :, :, 1::2], x02[:, :, :, 1::2]
        return torch.cat([x1+x2+x3+x4, -x1-x2+x3+x4, -x1+x2-x3+x4, x1-x2-x3+x4], 1)

class HaarIDWT(nn.Module):
    def forward(self, x):
        B, C4, H, W = x.shape
        C = C4 // 4
        x_LL, x_HL, x_LH, x_HH = x[:, 0:C], x[:, C:2*C], x[:, 2*C:3*C], x[:, 3*C:4*C]
        x1, x2 = (x_LL-x_HL-x_LH+x_HH)/2, (x_LL-x_HL+x_LH-x_HH)/2
        x3, x4 = (x_LL+x_HL-x_LH-x_HH)/2, (x_LL+x_HL+x_LH+x_HH)/2
        out = torch.zeros(B, C, H*2, W*2).to(x.device)
        out[:, :, 0::2, 0::2], out[:, :, 1::2, 0::2] = x1, x2
        out[:, :, 0::2, 1::2], out[:, :, 1::2, 1::2] = x3, x4
        return out