# B-INN: Verifying Image Ownership with Invertible Neural Network and Blockchain-based Verifiable Credential

This repository contains the implementation of the B-INN framework, which integrates an Invertible Neural Network (INN) with blockchain-supported Verifiable Credentials (VC) for digital image ownership verification.

## Project Structure

```
B-INN_2026/
├── config.py                  # Hyperparameters and settings
├── model.py                   # INN, U-Net, attack layer, full model
├── losses.py                  # Loss functions 
├── run.py                     # Training and evaluation pipeline
├── utils.py                   # ECC, DWT/IDWT, logging utilities
├── contracts/
│   ├── registration.sol   # VC issuance and revocation
│   └── verification.sol   # Ownership registration and verification
├── blockchain/
│   ├── blockchain_connector.py    # Web3 connection to Hyperledger Besu
│   ├── contract_handler.py        # Python interface for smart contract operations
│   └── __init__.py
└── README.md
```

## Dataset Information
The experiments use two publicly available datasets:

- **COCO Dataset**: https://cocodataset.org
- **Open Images Dataset V7**: https://storage.googleapis.com/openimages/web/index.html

## Requirements
- Python >= 3.8
- PyTorch >= 1.12
- FrEIA (Framework for Easily Invertible Architectures)
- Kornia >= 0.6
- torchvision
- numpy
- pandas
- openpyxl
- Pillow
- tqdm
- [Hyperledger Besu](https://besu.hyperledger.org)
- Solidity ^0.8.19
- web3.py >= 6.0

## Usage
### Training

The training script runs an ablation study with four configurations: INN-Only, INN+U-Net, INN+ECC, and the full proposed model (INN+ECC+U-Net).

```bash
python run.py --train_path ./data/train --val_path ./data/val --epochs 200 --batch_size 8
```

Training checkpoints are saved to `./checkpoints/`. Evaluation logs are stored as `.xlsx` files in the working directory.

### Blockchain Setup

1. Start a Hyperledger Besu node (default RPC at `http://127.0.0.1:8545`).
2. Compile and deploy `registration.sol` and `verification.sol` using a Solidity compiler (solc >= 0.8.19).
3. Save the contract ABIs as `contracts/registration_abi.json` and `contracts/verification_abi.json`.
4. Use `blockchain/contract_handler.py` to interact with the deployed contracts from Python.

### Smart Contract Test
You can easily test and deploy the smart contracts using the [Remix IDE](https://remix.ethereum.org/).

## License

This project is provided for academic and research purposes.
