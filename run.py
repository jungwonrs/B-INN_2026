import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, RandomSampler, Subset
import torchvision.transforms as T
import torch.nn.functional as F
from PIL import Image
import os, glob, tqdm, time, argparse
import numpy as np
from datetime import datetime
import kornia.augmentation as K
import kornia.geometry.transform as KGT
import kornia.filters 
from kornia.metrics import ssim as kornia_ssim

from config import Config
from model import RobustWatermarkModel
from losses import HybridLoss
from utils import ExcelLogger, ECC 

class ImageDataset(Dataset):
    def __init__(self, path, size=256):
        self.files = sum([glob.glob(os.path.join(path, f"*.{ext}")) for ext in ["jpg", "png", "jpeg"]], [])
        self.tf = T.Compose([T.Resize((size, size)), T.ToTensor()])
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        try: return self.tf(Image.open(self.files[i]).convert("RGB"))
        except: return torch.rand(3, Config.IMAGE_SIZE, Config.IMAGE_SIZE)

def get_metrics(a, b):
    mse = torch.mean((a - b)**2, dim=[1,2,3])
    psnr = 10 * torch.log10(1.0 / (mse + 1e-8))
    ssim = torch.mean(kornia_ssim(a, b, 11, 1.0), dim=[1,2,3])
    return psnr, ssim

def apply_atk(img, t, p):
    B, C, H, W = img.shape
    if t == 'Noise': return torch.clamp(img + torch.randn_like(img)*p, 0, 1)
    if t == 'Blur': return kornia.filters.gaussian_blur2d(img, (9,9), (p,p))
    if t == 'Rot': return KGT.rotate(img, torch.ones(B, device=img.device)*p)
    if t == 'JPEG': return K.RandomJPEG(jpeg_quality=float(p), p=1.0)(img)
    return img

def train_phase(model, loader, opt, crit, ecc, rot):
    model.train()
    total_l = 0
    for img in tqdm.tqdm(loader, leave=False):
        img = img.to(Config.DEVICE)
        raw = torch.randint(0, 2, (img.size(0), Config.DATA_BITS)).float().to(Config.DEVICE)
        tgt = ecc.encode(raw) if Config.USE_ECC else raw
        opt.zero_grad()
        stego, _, cleaned, pred = model(img, tgt, rot)
        loss, _ = crit(img, stego, cleaned, pred, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        total_l += loss.item()
    return total_l / len(loader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="./data/train")
    parser.add_argument("--val_path", type=str, default="./data/val")
    args = parser.parse_args()

    Config.MESSAGE_LENGTH = Config.DATA_BITS * 3 if Config.USE_ECC else Config.DATA_BITS
    
    model = RobustWatermarkModel().to(Config.DEVICE)
    crit = HybridLoss().to(Config.DEVICE)
    ecc = ECC(repeats=3)
    
    params = [{'params': model.inn.parameters(), 'lr': Config.LR_INN}]
    if model.unet: params.append({'params': model.unet.parameters(), 'lr': Config.LR_DEFENSE})
    opt = optim.Adam(params, betas=Config.BETAS)

    train_ds = ImageDataset(args.train_path)
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4)

    for epoch in range(1, Config.EPOCHS + 1):
        rot = min(90.0, (epoch / 40) * 90.0)
        loss = train_phase(model, train_loader, opt, crit, ecc, rot)
        print(f"Epoch {epoch}/{Config.EPOCHS} - Loss: {loss:.4f}")
        
        if epoch % 20 == 0:
            torch.save(model.state_dict(), f"{Config.CHECKPOINT_DIR}/model_ep{epoch}.pth")

if __name__ == "__main__":
    main()