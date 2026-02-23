import torch

class Config:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 42
    IMAGE_SIZE = 256
    CHANNELS = 3
    BATCH_SIZE = 8
    DATA_BITS = 256 
    MESSAGE_LENGTH = 256 
    EPOCHS = 160
    LR_INN = 1e-4
    LR_DEFENSE = 2e-4
    BETAS = (0.5, 0.999)
    LAMBDA_QUALITY = 50.0       
    LAMBDA_RESTORATION = 5.0     
    LAMBDA_BITS = 2.0            
    BLOCK_SIZE = 16
    SUBNET_WIDTH = 16
    PROB_IDENTITY = 0.2
    PROB_NOISE = 0.3
    PROB_BLUR = 0.3
    PROB_ROTATION = 0.2
    USE_STN = False
    USE_UNET = True  
    USE_ECC = True   
    CHECKPOINT_DIR = "./checkpoints"