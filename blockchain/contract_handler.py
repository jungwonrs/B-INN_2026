import hashlib
import json
import os
import torch
import numpy as np
from blockchain.blockchain_connector import BlockchainConnector


class ContractHandler:
    def __init__(self, private_key, reg_address=None, ver_address=None, connector=None):
        self.connector = connector or BlockchainConnector()
        self.private_key = private_key
        self.account = self.connector.get_account(private_key)

        reg_abi = self._load_abi("contracts/registration_abi.json")
        ver_abi = self._load_abi("contracts/verification_abi.json")

        if reg_address:
            self.reg_contract = self.connector.get_contract(reg_address, reg_abi)
        if ver_address:
            self.ver_contract = self.connector.get_contract(ver_address, ver_abi)

    def _load_abi(self, path):
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        return []

    @staticmethod
    def compute_image_hash(image_tensor):
        if isinstance(image_tensor, torch.Tensor):
            data = image_tensor.cpu().numpy().tobytes()
        else:
            data = image_tensor.tobytes()
        return hashlib.sha256(data).digest()

    @staticmethod
    def compute_vc_hash(vc_data):
        if isinstance(vc_data, dict):
            vc_str = json.dumps(vc_data, sort_keys=True)
        else:
            vc_str = str(vc_data)
        return hashlib.sha256(vc_str.encode()).digest()

    @staticmethod
    def hash_to_bits(hash_bytes, length=256):
        bit_str = bin(int.from_bytes(hash_bytes, 'big'))[2:].zfill(256)
        bits = [int(b) for b in bit_str[:length]]
        return torch.tensor(bits, dtype=torch.float32)

    def issue_vc(self, expired_at, secret_key_hash, image_hash, auth_threshold, signature):
        func = self.reg_contract.functions.issueVC(
            expired_at,
            secret_key_hash,
            image_hash,
            auth_threshold,
            signature
        )
        receipt = self.connector.send_transaction(func, self.private_key)
        logs = self.reg_contract.events.VCIssued().process_receipt(receipt)
        if logs:
            return logs[0]['args']['vcId']
        return None

    def get_vc(self, vc_id):
        func = self.reg_contract.functions.callVC(vc_id)
        return self.connector.call_function(func)

    def revoke_vc(self, vc_id):
        func = self.reg_contract.functions.revokeVC(vc_id)
        return self.connector.send_transaction(func, self.private_key)

    def get_vc_hash_onchain(self, vc_id):
        func = self.reg_contract.functions.getVCHash(vc_id)
        return self.connector.call_function(func)

    def register_ownership(self, image_hash, vc_hash, claim_image_hash):
        func = self.ver_contract.functions.registerOwnership(
            image_hash, vc_hash, claim_image_hash
        )
        return self.connector.send_transaction(func, self.private_key)

    def verify_ownership(self, image_hash, extracted_message, bit_accuracy, auth_threshold):
        accuracy_scaled = int(bit_accuracy * 10000)
        threshold_scaled = int(auth_threshold * 10000)

        func = self.ver_contract.functions.verifyOwnership(
            image_hash, extracted_message, accuracy_scaled, threshold_scaled
        )
        return self.connector.send_transaction(func, self.private_key)

    def compute_bit_accuracy(self, extracted_bits, original_hash_bits):
        if isinstance(extracted_bits, torch.Tensor):
            extracted_bits = extracted_bits.cpu()
        if isinstance(original_hash_bits, torch.Tensor):
            original_hash_bits = original_hash_bits.cpu()
        matching = (extracted_bits == original_hash_bits).float().mean().item()
        return matching
