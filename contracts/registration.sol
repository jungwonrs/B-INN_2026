pragma solidity ^0.8.19;

contract registration {
    uint256 public constant THRESHOLD_SCALE = 10000;

    struct VerifiableCredential {
        bytes32 id;
        address owner;
        uint256 issuedAt;
        uint256 expiredAt;
        bytes32 imageHash;
        uint256 authThreshold;
        bytes signature;
        bool revoked;
    }

    mapping(bytes32 => VerifiableCredential) private userVC;
    mapping(bytes32 => bool) public revocationList;

    event VCIssued(bytes32 indexed vcId, address indexed owner, bytes32 indexed imageHash);
    event VCRevoked(bytes32 indexed vcId, address indexed owner);

    modifier vcExists(bytes32 vcId) {
        require(userVC[vcId].owner != address(0), "VC does not exist");
        _;
    }

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
        require(expiredAt > block.timestamp, "Expiration must be in the future");
        require(authThreshold <= THRESHOLD_SCALE, "Threshold exceeds scale");

        bytes32 vcId = sha256(abi.encode(
            msg.sender,
            secretKeyHash,
            imageHash,
            block.timestamp,
            block.chainid
        ));
        require(userVC[vcId].owner == address(0), "VC already exists");

        userVC[vcId] = VerifiableCredential({
            id: vcId,
            owner: msg.sender,
            issuedAt: block.timestamp,
            expiredAt: expiredAt,
            imageHash: imageHash,
            authThreshold: authThreshold,
            signature: signature,
            revoked: false
        });

        emit VCIssued(vcId, msg.sender, imageHash);
        return vcId;
    }

    function callVC(bytes32 vcId) external view vcExists(vcId) onlyOwnerOf(vcId) returns (
        bytes32 id,
        address owner,
        uint256 issuedAt,
        uint256 expiredAt,
        bytes32 imageHash,
        uint256 authThreshold,
        string memory status
    ) {
        VerifiableCredential storage vc = userVC[vcId];
        status = isValidVC(vcId) ? "activate" : "inactivate";
        return (vc.id, vc.owner, vc.issuedAt, vc.expiredAt, vc.imageHash, vc.authThreshold, status);
    }

    function revokeVC(bytes32 vcId) external vcExists(vcId) onlyOwnerOf(vcId) {
        VerifiableCredential storage vc = userVC[vcId];
        require(!vc.revoked, "Already revoked");
        vc.revoked = true;
        revocationList[vcId] = true;
        emit VCRevoked(vcId, msg.sender);
    }

    function isValidVC(bytes32 vcId) public view returns (bool) {
        VerifiableCredential storage vc = userVC[vcId];
        return vc.owner != address(0) && !vc.revoked && block.timestamp <= vc.expiredAt;
    }

    function getVCHash(bytes32 vcId) public view vcExists(vcId) returns (bytes32) {
        VerifiableCredential storage vc = userVC[vcId];
        return sha256(abi.encode(
            vc.id,
            vc.owner,
            vc.issuedAt,
            vc.expiredAt,
            vc.imageHash,
            vc.authThreshold,
            vc.signature
        ));
    }

    function getVCBinding(bytes32 vcId) external view vcExists(vcId) returns (
        bytes32 vcHash,
        address owner,
        bytes32 imageHash,
        uint256 authThreshold,
        uint256 issuedAt,
        uint256 expiredAt,
        bool active
    ) {
        VerifiableCredential storage vc = userVC[vcId];
        return (
            getVCHash(vcId),
            vc.owner,
            vc.imageHash,
            vc.authThreshold,
            vc.issuedAt,
            vc.expiredAt,
            isValidVC(vcId)
        );
    }
}
