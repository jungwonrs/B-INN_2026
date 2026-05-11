import torch
import torch.nn as nn
from config import Config


class HybridLoss(nn.Module):
    def __init__(self):
        super(HybridLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.w_qual = Config.LAMBDA_QUALITY
        self.w_rest = Config.LAMBDA_RESTORATION
        self.w_bits = Config.LAMBDA_BITS

    def forward(self, original_img, stego_img, cleaned_img, pred_bits, target_bits):
        loss_quality = self.mse_loss(stego_img, original_img)
        loss_restoration = self.l1_loss(cleaned_img, stego_img)
        loss_bits = self.bce_loss(pred_bits, target_bits)

        total_loss = (self.w_qual * loss_quality) + \
                     (self.w_rest * loss_restoration) + \
                     (self.w_bits * loss_bits)

        log_dict = {
            "loss_quality": loss_quality.item(),
            "loss_restoration": loss_restoration.item(),
            "loss_bits": loss_bits.item(),
            "total_loss": total_loss.item()
        }
        return total_loss, log_dict
