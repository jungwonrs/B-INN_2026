// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

contract registration {

    struct VerifiableCredential {
        bytes32 id;
        address owner;
        uint256 issuedAt;
        uint256 expiredAt;
        bytes32 imageHash;
        uint256 authThreshold;
        bytes signature;
        string status;
    }

    mapping(bytes32 => VerifiableCredential) private userVC;
    mapping(bytes32 => bool) public revocationList;

    event VCIssued(bytes32 indexed vcId, address indexed owner);
    event VCRevoked(bytes32 indexed vcId, address indexed owner);

    modifier onlyOwnerOf(bytes32 vcId) {
        require(userVC[vcId].owner == msg.sender, "Not credential owner");
        _;
    }

    function issueVC(
        uint256 expiredAt,
        bytes32 secretKeyHash,
        bytes32 imageHash,
        uint256 authThreshold,
        bytes calldata signature
    ) external returns (bytes32) {
        bytes32 vcId = keccak256(abi.encodePacked(msg.sender, secretKeyHash));
        require(userVC[vcId].owner == address(0), "VC already exists");

        userVC[vcId] = VerifiableCredential({
            id: vcId,
            owner: msg.sender,
            issuedAt: block.timestamp,
            expiredAt: expiredAt,
            imageHash: imageHash,
            authThreshold: authThreshold,
            signature: signature,
            status: "activate"
        });

        emit VCIssued(vcId, msg.sender);
        return vcId;
    }

    function callVC(bytes32 vcId) external view onlyOwnerOf(vcId) returns (
        bytes32 id,
        address owner,
        uint256 issuedAt,
        uint256 expiredAt,
        bytes32 imageHash,
        uint256 authThreshold,
        string memory status
    ) {
        VerifiableCredential storage vc = userVC[vcId];
        require(keccak256(bytes(vc.status)) == keccak256(bytes("activate")), "VC not active");
        return (vc.id, vc.owner, vc.issuedAt, vc.expiredAt, vc.imageHash, vc.authThreshold, vc.status);
    }

    function revokeVC(bytes32 vcId) external onlyOwnerOf(vcId) {
        VerifiableCredential storage vc = userVC[vcId];
        require(keccak256(bytes(vc.status)) == keccak256(bytes("activate")), "Already inactive");
        vc.status = "inactivate";
        revocationList[vcId] = true;
        emit VCRevoked(vcId, msg.sender);
    }

    function getVCHash(bytes32 vcId) external view returns (bytes32) {
        VerifiableCredential storage vc = userVC[vcId];
        return keccak256(abi.encodePacked(
            vc.id, vc.owner, vc.issuedAt, vc.expiredAt,
            vc.imageHash, vc.authThreshold
        ));
    }
}
