import torch
import torch.nn as nn
import hashlib
import pandas as pd
import os


class ECC:
    def __init__(self, repeats=3):
        self.repeats = repeats

    def encode(self, data_bits):
        return data_bits.repeat_interleave(self.repeats, dim=1)

    def decode(self, pred_logits):
        B, total_bits = pred_logits.shape
        original_len = total_bits // self.repeats
        reshaped = pred_logits.view(B, original_len, self.repeats)
        summed = reshaped.sum(dim=2)
        decoded = (summed > 0).float()
        return decoded


class ExcelLogger:
    def __init__(self, filename="training_log.xlsx"):
        self.filename = filename
        self.df = None

    def log(self, epoch, metrics):
        new_row = {"Epoch": epoch}
        new_row.update(metrics)
        new_df = pd.DataFrame([new_row])
        try:
            if not os.path.exists(self.filename):
                self.df = new_df
                self.df.to_excel(self.filename, index=False)
            else:
                self.df = pd.read_excel(self.filename)
                self.df = pd.concat([self.df, new_df], ignore_index=True)
                self.df.to_excel(self.filename, index=False)
        except Exception as e:
            print(f"Log save error: {e}")


def string_to_bits(s, length):
    hash_obj = hashlib.sha256(s.encode())
    hex_dig = hash_obj.hexdigest()
    binary_str = bin(int(hex_dig, 16))[2:].zfill(256)
    bits = [int(b) for b in binary_str[:length]]
    return torch.tensor(bits, dtype=torch.float32).unsqueeze(0)


class HaarDWT(nn.Module):
    def __init__(self):
        super(HaarDWT, self).__init__()

    def forward(self, x):
        x01 = x[:, :, 0::2, :] / 2
        x02 = x[:, :, 1::2, :] / 2
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]
        x_LL = x1 + x2 + x3 + x4
        x_HL = -x1 - x2 + x3 + x4
        x_LH = -x1 + x2 - x3 + x4
        x_HH = x1 - x2 - x3 + x4
        return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)


class HaarIDWT(nn.Module):
    def __init__(self):
        super(HaarIDWT, self).__init__()

    def forward(self, x):
        B, C_total, H, W = x.shape
        C = C_total // 4
        x_LL = x[:, 0:C, :, :]
        x_HL = x[:, C:2*C, :, :]
        x_LH = x[:, 2*C:3*C, :, :]
        x_HH = x[:, 3*C:4*C, :, :]
        x1 = (x_LL - x_HL - x_LH + x_HH) / 2
        x2 = (x_LL - x_HL + x_LH - x_HH) / 2
        x3 = (x_LL + x_HL - x_LH - x_HH) / 2
        x4 = (x_LL + x_HL + x_LH + x_HH) / 2
        x_out = torch.zeros(B, C, H * 2, W * 2).to(x.device)
        x_out[:, :, 0::2, 0::2] = x1
        x_out[:, :, 1::2, 0::2] = x2
        x_out[:, :, 0::2, 1::2] = x3
        x_out[:, :, 1::2, 1::2] = x4
        return x_out
