import torch
import torch.nn as nn
import torch.nn.functional as F
import FrEIA.framework as Ff
import FrEIA.modules as Fm
import kornia.augmentation as K
import kornia.geometry.transform as KGT

from config import Config
from utils import HaarDWT, HaarIDWT


class DenseSubnet(nn.Module):
    def __init__(self, c_in, c_out):
        super(DenseSubnet, self).__init__()
        self.width = Config.SUBNET_WIDTH
        self.conv1 = nn.Conv2d(c_in, self.width, 3, padding=1)
        self.conv2 = nn.Conv2d(c_in + self.width, self.width, 3, padding=1)
        self.conv3 = nn.Conv2d(c_in + 2 * self.width, self.width, 3, padding=1)
        self.conv4 = nn.Conv2d(c_in + 3 * self.width, self.width, 3, padding=1)
        self.final_conv = nn.Conv2d(c_in + 4 * self.width, c_out, 3, padding=1)
        self.final_conv.weight.data.zero_()
        self.final_conv.bias.data.zero_()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        cat1 = torch.cat([x, self.relu(self.conv1(x))], dim=1)
        cat2 = torch.cat([cat1, self.relu(self.conv2(cat1))], dim=1)
        cat3 = torch.cat([cat2, self.relu(self.conv3(cat2))], dim=1)
        cat4 = torch.cat([cat3, self.relu(self.conv4(cat3))], dim=1)
        return self.final_conv(cat4)


class WatermarkINN(nn.Module):
    def __init__(self):
        super(WatermarkINN, self).__init__()
        self.c_in = Config.CHANNELS * 4
        self.inn = Ff.SequenceINN(self.c_in, Config.IMAGE_SIZE // 2, Config.IMAGE_SIZE // 2)
        for _ in range(Config.BLOCK_SIZE):
            self.inn.append(Fm.AllInOneBlock, subnet_constructor=DenseSubnet)

        self.msg_len = Config.MESSAGE_LENGTH
        self.msg_expander = nn.Linear(self.msg_len, (Config.IMAGE_SIZE // 2) ** 2)
        self.msg_compressor = nn.Linear((Config.IMAGE_SIZE // 2) ** 2, self.msg_len)

    def embed(self, x_dwt, message):
        z, _ = self.inn(x_dwt)
        msg_plane = self.msg_expander(message).view(-1, 1, Config.IMAGE_SIZE // 2, Config.IMAGE_SIZE // 2)
        z_stego = torch.cat([z[:, :-1, :, :], msg_plane], dim=1)
        stego_dwt, _ = self.inn(z_stego, rev=True)
        return stego_dwt

    def extract(self, stego_dwt):
        z_rec, _ = self.inn(stego_dwt)
        msg_plane_rec = z_rec[:, -1:, :, :].contiguous()
        msg_flat = msg_plane_rec.view(msg_plane_rec.size(0), -1)
        return self.msg_compressor(msg_flat)


class DifferentiableAttackLayer(nn.Module):
    def __init__(self):
        super(DifferentiableAttackLayer, self).__init__()
        self.aug = K.AugmentationSequential(
            K.RandomGaussianNoise(mean=0.0, std=0.05, p=Config.PROB_NOISE),
            K.RandomGaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0), p=Config.PROB_BLUR),
            K.RandomResizedCrop(size=(Config.IMAGE_SIZE, Config.IMAGE_SIZE), scale=(0.8, 1.0), p=Config.PROB_ROTATION),
            same_on_batch=False
        )

    def forward(self, x, max_rot_degrees=0.0):
        if self.training:
            x = self.aug(x)
            if torch.rand(1).item() < 0.5:
                try:
                    x = K.RandomJPEG(jpeg_quality=50.0, p=1.0)(x)
                except:
                    x = x + (torch.randn_like(x) * 0.02)
            if max_rot_degrees > 0 and torch.rand(1).item() < Config.PROB_ROTATION:
                angles = (torch.rand(x.size(0), device=x.device) * 2 - 1) * max_rot_degrees
                x = KGT.rotate(x, angles)
        return x


class DefenseUNet(nn.Module):
    def __init__(self):
        super(DefenseUNet, self).__init__()
        self.enc1 = self._block(Config.CHANNELS, 32)
        self.enc2 = self._block(32, 64)
        self.enc3 = self._block(64, 128)
        self.enc4 = self._block(128, 256)
        self.enc5 = self._block(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = self._block(512, 1024)
        self.upconv5 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec5 = self._block(1024, 512)
        self.upconv4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = self._block(512, 256)
        self.upconv3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = self._block(256, 128)
        self.upconv2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = self._block(128, 64)
        self.upconv1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = self._block(64, 32)
        self.final = nn.Conv2d(32, Config.CHANNELS, kernel_size=1)

    def _block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))
        b = self.bottleneck(self.pool(e5))
        d5 = self.dec5(torch.cat((self.upconv5(b), e5), dim=1))
        d4 = self.dec4(torch.cat((self.upconv4(d5), e4), dim=1))
        d3 = self.dec3(torch.cat((self.upconv3(d4), e3), dim=1))
        d2 = self.dec2(torch.cat((self.upconv2(d3), e2), dim=1))
        d1 = self.dec1(torch.cat((self.upconv1(d2), e1), dim=1))
        return x + self.final(d1)


class RobustWatermarkModel(nn.Module):
    def __init__(self):
        super(RobustWatermarkModel, self).__init__()
        self.dwt = HaarDWT()
        self.idwt = HaarIDWT()
        self.inn = WatermarkINN()
        self.attacker = DifferentiableAttackLayer()
        self.unet = DefenseUNet() if Config.USE_UNET else None

    def forward(self, x, message, current_max_rot=0.0):
        x_dwt = self.dwt(x)
        stego_dwt = self.inn.embed(x_dwt, message)
        stego_img = torch.clamp(self.idwt(stego_dwt), 0, 1)

        attacked_img = self.attacker(stego_img, max_rot_degrees=current_max_rot)

        if self.unet is not None:
            cleaned_img = self.unet(attacked_img)
        else:
            cleaned_img = attacked_img

        cleaned_dwt = self.dwt(cleaned_img)
        pred_bits = self.inn.extract(cleaned_dwt)

        return stego_img, attacked_img, cleaned_img, pred_bits
