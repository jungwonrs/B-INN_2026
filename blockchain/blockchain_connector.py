from web3 import Web3
from config import Config


class BlockchainConnector:
    def __init__(self, rpc_url=None, chain_id=None):
        self.rpc_url = rpc_url or Config.BLOCKCHAIN_RPC
        self.chain_id = chain_id or Config.CHAIN_ID
        self.w3 = Web3(Web3.HTTPProvider(self.rpc_url))

    def is_connected(self):
        return self.w3.is_connected()

    def get_account(self, private_key):
        account = self.w3.eth.account.from_key(private_key)
        return account

    def deploy_contract(self, abi, bytecode, private_key):
        account = self.get_account(private_key)
        contract = self.w3.eth.contract(abi=abi, bytecode=bytecode)

        tx = contract.constructor().build_transaction({
            'from': account.address,
            'nonce': self.w3.eth.get_transaction_count(account.address),
            'gas': 3000000,
            'gasPrice': self.w3.eth.gas_price,
            'chainId': self.chain_id
        })

        signed = self.w3.eth.account.sign_transaction(tx, private_key)
        tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        return receipt.contractAddress

    def get_contract(self, address, abi):
        return self.w3.eth.contract(address=address, abi=abi)

    def send_transaction(self, contract_func, private_key):
        account = self.get_account(private_key)
        tx = contract_func.build_transaction({
            'from': account.address,
            'nonce': self.w3.eth.get_transaction_count(account.address),
            'gas': 3000000,
            'gasPrice': self.w3.eth.gas_price,
            'chainId': self.chain_id
        })
        signed = self.w3.eth.account.sign_transaction(tx, private_key)
        tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        return receipt

    def call_function(self, contract_func):
        return contract_func.call()
