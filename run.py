import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, RandomSampler, Subset
import torchvision.transforms as T
import torch.nn.functional as F
from PIL import Image
import os
import glob
import tqdm
import numpy as np
import time
import argparse
from datetime import datetime

import kornia.augmentation as K
import kornia.geometry.transform as KGT
import kornia.filters
from kornia.metrics import ssim as kornia_ssim

from config import Config
from model import RobustWatermarkModel
from losses import HybridLoss
from utils import ExcelLogger, ECC

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


class RealImageDataset(Dataset):
    def __init__(self, root_dir, image_size=256):
        self.root_dir = root_dir
        self.files = glob.glob(os.path.join(root_dir, "*.jpg")) + \
                     glob.glob(os.path.join(root_dir, "*.png")) + \
                     glob.glob(os.path.join(root_dir, "*.jpeg"))
        if len(self.files) == 0:
            print(f"[Warning] No images found in {root_dir}")
        self.transform = T.Compose([T.Resize((image_size, image_size)), T.ToTensor()])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.files[idx]).convert("RGB")
            return self.transform(img)
        except:
            return torch.rand(3, Config.IMAGE_SIZE, Config.IMAGE_SIZE)


def calculate_nc(pred_bits, target_bits):
    pred_probs = torch.sigmoid(pred_bits)
    pred_binary = (pred_probs > 0.5).float()
    pred_vec = pred_binary.view(pred_binary.size(0), -1) * 2 - 1
    target_vec = target_bits.view(target_bits.size(0), -1) * 2 - 1
    dot_product = torch.sum(pred_vec * target_vec, dim=1)
    norm_prod = torch.norm(pred_vec, dim=1) * torch.norm(target_vec, dim=1)
    return dot_product / (norm_prod + 1e-8)


def get_psnr_batch(img1, img2):
    mse = torch.mean((img1 - img2) ** 2, dim=[1, 2, 3])
    return 10 * torch.log10(1.0 / (mse + 1e-8))


def get_ssim_batch(img1, img2):
    ssim_map = kornia_ssim(img1, img2, window_size=11, max_val=1.0)
    return torch.mean(ssim_map, dim=[1, 2, 3])


def apply_specific_attack(images, attack_type, param):
    B, C, H, W = images.shape
    if attack_type == 'Identity':
        return images
    elif attack_type == 'Noise':
        noise = torch.randn_like(images) * param
        return torch.clamp(images + noise, 0, 1)
    elif attack_type == 'Blur':
        return kornia.filters.gaussian_blur2d(images, (9, 9), (param, param))
    elif attack_type == 'Rot':
        angle = torch.ones(B, device=images.device) * param
        return KGT.rotate(images, angle)
    elif attack_type == 'Crop':
        op = K.RandomResizedCrop(size=(H, W), scale=(param, param), ratio=(1.0, 1.0), p=1.0)
        return op(images)
    elif attack_type == 'Resize':
        h_new, w_new = int(H * param), int(W * param)
        resized = F.interpolate(images, size=(h_new, w_new), mode='bilinear', align_corners=False)
        return F.interpolate(resized, size=(H, W), mode='bilinear', align_corners=False)
    elif attack_type == 'JPEG':
        op = K.RandomJPEG(jpeg_quality=float(param), p=1.0)
        return op(images)
    return images


def run_robustness_test(model, val_loader, device, epoch, train_stats, logger, elapsed_time):
    ecc = ECC(repeats=3)

    psnr_list, ssim_list = [], []
    with torch.no_grad():
        for images in val_loader:
            images = images.to(device)
            B = images.size(0)
            raw_data = torch.randint(0, 2, (B, Config.DATA_BITS)).float().to(device)
            if Config.USE_ECC:
                target = ecc.encode(raw_data)
            else:
                target = raw_data
            x_dwt = model.dwt(images)
            stego_dwt = model.inn.embed(x_dwt, target)
            stego_img = torch.clamp(model.idwt(stego_dwt), 0, 1)
            psnr_list.extend(get_psnr_batch(images, stego_img).cpu().numpy())
            ssim_list.extend(get_ssim_batch(images, stego_img).cpu().numpy())

    stego_psnr_mean = np.mean(psnr_list)
    stego_ssim_mean = np.mean(ssim_list)

    scenarios = [('Identity', 0)]
    scenarios += [('JPEG', q) for q in [10, 30, 50, 70, 90]]
    scenarios += [('Blur', sig) for sig in [0.5, 1.0, 2.0, 3.0, 4.0]]
    scenarios += [('Noise', sig) for sig in [0.01, 0.05, 0.1, 0.2]]
    scenarios += [('Crop', s) for s in [0.5, 0.6, 0.7, 0.8, 0.9]]
    scenarios += [('Rot', d) for d in [5, 15, 30, 45, 60]]
    scenarios += [('Resize', s) for s in [0.3, 0.5, 0.8, 1.2, 1.5, 2.0]]

    identity_acc = 0.0

    for (atk_name, param) in scenarios:
        acc_list, nc_list = [], []
        try:
            with torch.no_grad():
                for images in val_loader:
                    images = images.to(device)
                    B = images.size(0)
                    raw_data = torch.randint(0, 2, (B, Config.DATA_BITS)).float().to(device)
                    if Config.USE_ECC:
                        target = ecc.encode(raw_data)
                    else:
                        target = raw_data
                    x_dwt = model.dwt(images)
                    stego_dwt = model.inn.embed(x_dwt, target)
                    stego_img = torch.clamp(model.idwt(stego_dwt), 0, 1)

                    attacked_img = apply_specific_attack(stego_img, atk_name, param)

                    if model.unet is not None:
                        cleaned = model.unet(attacked_img)
                    else:
                        cleaned = attacked_img

                    cleaned_dwt = model.dwt(cleaned)
                    pred_bits = model.inn.extract(cleaned_dwt)

                    if Config.USE_ECC:
                        decoded_bits = ecc.decode(pred_bits)
                        acc_batch = (decoded_bits == raw_data).float().mean().item()
                    else:
                        decoded_bits = (torch.sigmoid(pred_bits) > 0.5).float()
                        acc_batch = (decoded_bits == raw_data).float().mean().item()

                    nc_batch = calculate_nc(pred_bits, target).mean().item()
                    acc_list.append(acc_batch)
                    nc_list.append(nc_batch)

            acc_mean = np.mean(acc_list)
            nc_mean = np.mean(nc_list)
            if atk_name == 'Identity':
                identity_acc = acc_mean

            row_data = {
                "Epoch": epoch,
                "Attack_Type": atk_name,
                "Attack_Param": param,
                "Bit_Acc_Mean": acc_mean,
                "NC_Mean": nc_mean,
                "Stego_PSNR": stego_psnr_mean,
                "Stego_SSIM": stego_ssim_mean,
                "W_Quality": Config.LAMBDA_QUALITY,
                "W_Restoration": Config.LAMBDA_RESTORATION,
                "W_Bits": Config.LAMBDA_BITS,
                "Block_Size": Config.BLOCK_SIZE,
                "Use_UNet": Config.USE_UNET,
                "Use_ECC": Config.USE_ECC,
                "Elapsed_Time_Sec": elapsed_time,
                **train_stats
            }
            logger.log(epoch, row_data)

        except Exception as e:
            print(f"      [Error] {atk_name}_{param}: {e}")

    return identity_acc


def run_experiment(exp_name, config_overrides, train_dir, val_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_name = f"{exp_name}_{timestamp}"
    print(f"\n=== [Experiment] {unique_name} ===")

    for key, value in config_overrides.items():
        setattr(Config, key, value)

    if Config.USE_ECC:
        Config.MESSAGE_LENGTH = Config.DATA_BITS * 3
    else:
        Config.MESSAGE_LENGTH = Config.DATA_BITS

    print(f"  UNet={Config.USE_UNET}, ECC={Config.USE_ECC}, MsgLen={Config.MESSAGE_LENGTH}")

    logger = ExcelLogger(filename=f"log_{unique_name}.xlsx")
    model = RobustWatermarkModel().to(Config.DEVICE)
    criterion = HybridLoss().to(Config.DEVICE)
    ecc = ECC(repeats=3)

    params_to_update = [{'params': model.inn.parameters(), 'lr': Config.LR_INN}]
    if model.unet is not None:
        params_to_update.append({'params': model.unet.parameters(), 'lr': Config.LR_DEFENSE})
    optimizer = optim.Adam(params_to_update, betas=Config.BETAS)

    full_train_dataset = RealImageDataset(train_dir, image_size=Config.IMAGE_SIZE)
    full_val_dataset = RealImageDataset(val_dir, image_size=Config.IMAGE_SIZE)

    POOL_SIZE = 3000
    TARGET_ITERS = 3000
    val_sampler = RandomSampler(full_val_dataset, replacement=True, num_samples=100)
    val_loader = DataLoader(full_val_dataset, batch_size=Config.BATCH_SIZE, sampler=val_sampler, num_workers=0)

    best_avg_acc = 0.0
    VAL_INTERVAL = 20
    total_start = time.time()

    for epoch in range(1, Config.EPOCHS + 1):
        epoch_start = time.time()
        RAMP_UP_EPOCHS = 40
        MAX_ROTATION = 90.0
        current_max_rot = min(MAX_ROTATION, (epoch / RAMP_UP_EPOCHS) * MAX_ROTATION)

        indices = torch.randperm(len(full_train_dataset))[:POOL_SIZE]
        subset = Subset(full_train_dataset, indices)
        train_loader = DataLoader(
            subset, batch_size=Config.BATCH_SIZE,
            sampler=RandomSampler(subset, replacement=True, num_samples=TARGET_ITERS * Config.BATCH_SIZE),
            num_workers=0, pin_memory=True
        )

        model.train()
        r_loss, r_qual, r_rest, r_bits = 0.0, 0.0, 0.0, 0.0
        pbar = tqdm.tqdm(train_loader, total=TARGET_ITERS, desc=f"Ep {epoch}", leave=False)

        for images in pbar:
            images = images.to(Config.DEVICE, non_blocking=True)
            B = images.size(0)

            raw_data = torch.randint(0, 2, (B, Config.DATA_BITS)).float().to(Config.DEVICE)
            if Config.USE_ECC:
                target_bits = ecc.encode(raw_data)
            else:
                target_bits = raw_data

            optimizer.zero_grad()
            stego, _, cleaned, pred_bits = model(images, target_bits, current_max_rot=current_max_rot)
            loss, l_dict = criterion(images, stego, cleaned, pred_bits, target_bits)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()

            r_loss += loss.item()
            r_qual += l_dict['loss_quality']
            r_rest += l_dict['loss_restoration']
            r_bits += l_dict['loss_bits']
            pbar.set_postfix({'L': f"{loss.item():.2f}", 'Rot': f"{current_max_rot:.1f}"})

        epoch_duration = time.time() - epoch_start
        train_stats = {
            "Train_Loss_Total": r_loss / TARGET_ITERS,
            "Train_Loss_Qual": r_qual / TARGET_ITERS,
            "Train_Loss_Rest": r_rest / TARGET_ITERS,
            "Train_Loss_Bits": r_bits / TARGET_ITERS,
            "Curriculum_Rot_Deg": current_max_rot
        }

        if epoch % VAL_INTERVAL == 0 or epoch == Config.EPOCHS:
            print(f"\n[Val] Epoch {epoch}: {epoch_duration:.1f}s | Rot: {current_max_rot:.1f}")
            identity_acc = run_robustness_test(
                model, val_loader, Config.DEVICE, epoch, train_stats, logger, epoch_duration
            )
            if identity_acc > best_avg_acc:
                best_avg_acc = identity_acc
                os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
                torch.save(model.state_dict(), f"{Config.CHECKPOINT_DIR}/best_{unique_name}.pth")
        else:
            row_data = {"Epoch": epoch, "Attack_Type": "Training_Only",
                        "Elapsed_Time_Sec": epoch_duration, **train_stats}
            logger.log(epoch, row_data)
            print(f" -> Epoch {epoch} ({epoch_duration:.1f}s) Rot={current_max_rot:.1f}")

    total_duration = time.time() - total_start
    print(f"=== Done. Total: {total_duration:.1f}s ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    Config.EPOCHS = args.epochs
    Config.BATCH_SIZE = args.batch_size

    experiments = [
        {"name": "INN_Only",
         "params": {"USE_UNET": False, "USE_ECC": False, "USE_STN": False}},
        {"name": "INN_UNet",
         "params": {"USE_UNET": True, "USE_ECC": False, "USE_STN": False}},
        {"name": "INN_ECC",
         "params": {"USE_UNET": False, "USE_ECC": True, "USE_STN": False}},
        {"name": "INN_ECC_UNet_Proposed",
         "params": {"USE_UNET": True, "USE_ECC": True, "USE_STN": False}},
    ]

    for exp in experiments:
        run_experiment(exp["name"], exp["params"], args.train_path, args.val_path)
