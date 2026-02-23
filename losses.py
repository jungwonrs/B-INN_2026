import torch
import torch.nn as nn
from config import Config

class HybridLoss(nn.Module):
    def __init__(self):
        super(HybridLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.bce = nn.BCEWithLogitsLoss()
        self.w_qual = Config.LAMBDA_QUALITY
        self.w_rest = Config.LAMBDA_RESTORATION
        self.w_bits = Config.LAMBDA_BITS

    def forward(self, original, stego, cleaned, pred_bits, target_bits):
        loss_q = self.mse(stego, original)
        loss_r = self.l1(cleaned, stego)
        loss_b = self.bce(pred_bits, target_bits)
        total = (self.w_qual * loss_q) + (self.w_rest * loss_r) + (self.w_bits * loss_b)
        
        return total, {
            "loss_quality": loss_q.item(),
            "loss_restoration": loss_r.item(),
            "loss_bits": loss_b.item(),
            "total_loss": total.item()
        }