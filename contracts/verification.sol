// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

contract verification {

    struct OwnershipRecord {
        bytes32 imageHash;
        bytes32 vcHash;
        address owner;
        uint256 registeredAt;
    }

    mapping(bytes32 => OwnershipRecord) public ownershipMap;

    event OwnershipRegistered(bytes32 indexed imageHash, address indexed owner);
    event OwnershipVerified(bytes32 indexed imageHash, address indexed claimant, bool result);

    function registerOwnership(
        bytes32 imageHash,
        bytes32 vcHash,
        bytes32 claimImageHash
    ) external {
        require(imageHash == claimImageHash, "Image hash mismatch with VC claim");
        require(ownershipMap[imageHash].owner == address(0), "Already registered");

        ownershipMap[imageHash] = OwnershipRecord({
            imageHash: imageHash,
            vcHash: vcHash,
            owner: msg.sender,
            registeredAt: block.timestamp
        });

        emit OwnershipRegistered(imageHash, msg.sender);
    }

    function verifyOwnership(
        bytes32 imageHash,
        bytes32 extractedMessage,
        uint256 bitAccuracy,
        uint256 authThreshold
    ) external returns (bool) {
        OwnershipRecord storage record = ownershipMap[imageHash];
        require(record.owner != address(0), "No ownership record found");

        bool verified = false;

        if (extractedMessage == record.vcHash) {
            verified = true;
        } else if (bitAccuracy >= authThreshold) {
            verified = true;
        }

        emit OwnershipVerified(imageHash, msg.sender, verified);
        return verified;
    }

    function getOwnership(bytes32 imageHash) external view returns (
        address owner,
        bytes32 vcHash,
        uint256 registeredAt
    ) {
        OwnershipRecord storage record = ownershipMap[imageHash];
        return (record.owner, record.vcHash, record.registeredAt);
    }
}
