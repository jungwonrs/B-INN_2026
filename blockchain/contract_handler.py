import hashlib
import json
import os

import torch

from blockchain.blockchain_connector import BlockchainConnector


class ContractHandler:
    THRESHOLD_SCALE = 10000

    def __init__(self, private_key, reg_address=None, ver_address=None, connector=None):
        self.connector = connector or BlockchainConnector()
        self.private_key = private_key
        self.account = self.connector.get_account(private_key)

        reg_abi = self._load_abi("contracts/registration_abi.json")
        ver_abi = self._load_abi("contracts/verification_abi.json")

        if reg_address is not None:
            self.reg_contract = self.connector.get_contract(reg_address, reg_abi)
        if ver_address is not None:
            self.ver_contract = self.connector.get_contract(ver_address, ver_abi)

    def _load_abi(self, path):
        if not os.path.exists(path):
            return []
        with open(path, "r") as f:
            return json.load(f)

    @staticmethod
    def _to_bytes32(value):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().flatten().round().int().tolist()
            return ContractHandler.bits_to_bytes32(value)

        if isinstance(value, (list, tuple)):
            return ContractHandler.bits_to_bytes32(value)

        if isinstance(value, str):
            if value.startswith("0x"):
                value = value[2:]
            raw = bytes.fromhex(value)
        else:
            raw = bytes(value)

        if len(raw) != 32:
            raise ValueError("bytes32 value must be exactly 32 bytes")
        return raw

    @staticmethod
    def compute_image_hash(image_tensor):
        if isinstance(image_tensor, torch.Tensor):
            data = image_tensor.detach().cpu().numpy().tobytes()
        else:
            data = image_tensor.tobytes()
        return hashlib.sha256(data).digest()

    @staticmethod
    def compute_vc_hash_offchain(vc_data):
        s = json.dumps(vc_data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(s.encode()).digest()

    @staticmethod
    def hash_to_bits(hash_bytes, length=256):
        raw = ContractHandler._to_bytes32(hash_bytes)
        bit_str = bin(int.from_bytes(raw, "big"))[2:].zfill(256)
        bits = [int(b) for b in bit_str[:length]]
        return torch.tensor(bits, dtype=torch.float32)

    @staticmethod
    def bits_to_bytes32(bits):
        bits = list(bits)
        if len(bits) != 256:
            raise ValueError("Expected 256 bits")
        bit_str = "".join("1" if int(b) else "0" for b in bits)
        return int(bit_str, 2).to_bytes(32, "big")

    @staticmethod
    def scale_threshold(threshold):
        if threshold < 0 or threshold > 1:
            raise ValueError("Threshold must be in [0, 1]")
        return int(round(threshold * ContractHandler.THRESHOLD_SCALE))

    def set_registration_contract(self, registration_address):
        f = self.ver_contract.functions.setRegistrationContract(registration_address)
        return self.connector.send_transaction(f, self.private_key)

    def issue_vc(self, expired_at, secret_key_hash, image_hash, auth_threshold, signature=b""):
        if isinstance(auth_threshold, float):
            auth_threshold = self.scale_threshold(auth_threshold)

        f = self.reg_contract.functions.issueVC(
            expired_at,
            self._to_bytes32(secret_key_hash),
            self._to_bytes32(image_hash),
            int(auth_threshold),
            signature,
        )
        receipt = self.connector.send_transaction(f, self.private_key)
        logs = self.reg_contract.events.VCIssued().process_receipt(receipt)
        if len(logs) == 0:
            return None
        return logs[0]["args"]["vcId"]

    def get_vc(self, vc_id):
        f = self.reg_contract.functions.callVC(self._to_bytes32(vc_id))
        return self.connector.call_function(f)

    def revoke_vc(self, vc_id):
        f = self.reg_contract.functions.revokeVC(self._to_bytes32(vc_id))
        return self.connector.send_transaction(f, self.private_key)

    def get_vc_hash_onchain(self, vc_id):
        f = self.reg_contract.functions.getVCHash(self._to_bytes32(vc_id))
        return self.connector.call_function(f)

    def get_vc_binding(self, vc_id):
        f = self.reg_contract.functions.getVCBinding(self._to_bytes32(vc_id))
        return self.connector.call_function(f)

    def register_ownership(self, vc_id):
        f = self.ver_contract.functions.registerOwnership(self._to_bytes32(vc_id))
        return self.connector.send_transaction(f, self.private_key)

    def verify_ownership(self, image_hash, extracted_hash):
        f = self.ver_contract.functions.verifyOwnership(
            self._to_bytes32(image_hash),
            self._to_bytes32(extracted_hash),
        )
        return self.connector.send_transaction(f, self.private_key)

    def check_ownership(self, image_hash, extracted_hash, claimant_address=None):
        if claimant_address is None:
            claimant_address = self.account.address
        f = self.ver_contract.functions.checkOwnership(
            self._to_bytes32(image_hash),
            self._to_bytes32(extracted_hash),
            claimant_address,
        )
        return self.connector.call_function(f)

    def compute_bit_accuracy(self, extracted_bits, original_hash_bits):
        if isinstance(extracted_bits, torch.Tensor):
            extracted_bits = extracted_bits.detach().cpu()
        if isinstance(original_hash_bits, torch.Tensor):
            original_hash_bits = original_hash_bits.detach().cpu()
        return (extracted_bits == original_hash_bits).float().mean().item()
