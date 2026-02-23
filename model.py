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
        w = Config.SUBNET_WIDTH 
        self.layers = nn.ModuleList([
            nn.Conv2d(c_in, w, 3, padding=1),
            nn.Conv2d(c_in + w, w, 3, padding=1),
            nn.Conv2d(c_in + 2 * w, w, 3, padding=1),
            nn.Conv2d(c_in + 3 * w, w, 3, padding=1)
        ])
        self.final = nn.Conv2d(c_in + 4 * w, c_out, 3, padding=1)
        self.final.weight.data.zero_()
        self.final.bias.data.zero_()

    def forward(self, x):
        features = x
        for layer in self.layers:
            out = F.relu(layer(features))
            features = torch.cat([features, out], dim=1)
        return self.final(features)

class WatermarkINN(nn.Module):
    def __init__(self):
        super(WatermarkINN, self).__init__()
        self.c_in = Config.CHANNELS * 4
        self.inn = Ff.SequenceINN(self.c_in, Config.IMAGE_SIZE // 2, Config.IMAGE_SIZE // 2)
        for _ in range(Config.BLOCK_SIZE):
            self.inn.append(Fm.AllInOneBlock, subnet_constructor=DenseSubnet)
        
        m_len = Config.MESSAGE_LENGTH
        spatial_dim = (Config.IMAGE_SIZE // 2) ** 2
        self.expander = nn.Linear(m_len, spatial_dim)
        self.compressor = nn.Linear(spatial_dim, m_len)

    def embed(self, x_dwt, msg):
        z, _ = self.inn(x_dwt) 
        msg_p = self.expander(msg).view(-1, 1, Config.IMAGE_SIZE // 2, Config.IMAGE_SIZE // 2)
        z_stego = torch.cat([z[:, :-1, :, :], msg_p], dim=1)
        stego_dwt, _ = self.inn(z_stego, rev=True)
        return stego_dwt
    
    def extract(self, stego_dwt):
        z_rec, _ = self.inn(stego_dwt)
        msg_p_rec = z_rec[:, -1:, :, :].contiguous()
        return self.compressor(msg_p_rec.view(msg_p_rec.size(0), -1))

class DifferentiableAttack(nn.Module):
    def __init__(self):
        super(DifferentiableAttack, self).__init__()
        self.aug = K.AugmentationSequential(
            K.RandomGaussianNoise(mean=0.0, std=0.05, p=Config.PROB_NOISE),
            K.RandomGaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0), p=Config.PROB_BLUR),
            K.RandomResizedCrop(size=(Config.IMAGE_SIZE, Config.IMAGE_SIZE), scale=(0.8, 1.0), p=Config.PROB_ROTATION),
            same_on_batch=False
        )

    def forward(self, x, rot=0.0):
        if self.training:
            x = self.aug(x)
            if torch.rand(1).item() < 0.5:
                try: x = K.RandomJPEG(jpeg_quality=50.0, p=1.0)(x)
                except: x = x + (torch.randn_like(x) * 0.02)
            if rot > 0 and torch.rand(1).item() < Config.PROB_ROTATION:
                angles = (torch.rand(x.size(0), device=x.device) * 2 - 1) * rot
                x = KGT.rotate(x, angles)
        return x

class DefenseUNet(nn.Module):
    def __init__(self):
        super(DefenseUNet, self).__init__()
        def block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(True),
                nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(True)
            )
        self.e1, self.e2 = block(Config.CHANNELS, 32), block(32, 64)
        self.e3, self.e4 = block(64, 128), block(128, 256)
        self.e5 = block(256, 512)
        self.p = nn.MaxPool2d(2)
        self.b = block(512, 1024)
        self.u5, self.u4 = nn.ConvTranspose2d(1024, 512, 2, 2), nn.ConvTranspose2d(512, 256, 2, 2)
        self.u3, self.u2 = nn.ConvTranspose2d(256, 128, 2, 2), nn.ConvTranspose2d(128, 64, 2, 2)
        self.u1 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.d5, self.d4 = block(1024, 512), block(512, 256)
        self.d3, self.d2 = block(256, 128), block(128, 64)
        self.d1 = block(64, 32)
        self.f = nn.Conv2d(32, Config.CHANNELS, 1)

    def forward(self, x):
        c1 = self.e1(x)
        c2 = self.e2(self.p(c1))
        c3 = self.e3(self.p(c2))
        c4 = self.e4(self.p(c3))
        c5 = self.e5(self.p(c4))
        m = self.b(self.p(c5))
        o5 = self.d5(torch.cat([self.u5(m), c5], 1))
        o4 = self.d4(torch.cat([self.u4(o5), c4], 1))
        o3 = self.d3(torch.cat([self.u3(o4), c3], 1))
        o2 = self.d2(torch.cat([self.u2(o3), c2], 1))
        o1 = self.d1(torch.cat([self.u1(o2), c1], 1))
        return x + self.f(o1)

class RobustWatermarkModel(nn.Module):
    def __init__(self):
        super(RobustWatermarkModel, self).__init__()
        self.dwt, self.idwt = HaarDWT(), HaarIDWT()
        self.inn = WatermarkINN()
        self.atk = DifferentiableAttack()
        self.unet = DefenseUNet() if Config.USE_UNET else None

    def forward(self, x, msg, rot=0.0):
        stego = torch.clamp(self.idwt(self.inn.embed(self.dwt(x), msg)), 0, 1)
        attacked = self.atk(stego, rot)
        cleaned = self.unet(attacked) if self.unet else attacked
        pred = self.inn.extract(self.dwt(cleaned))
        return stego, attacked, cleaned, pred