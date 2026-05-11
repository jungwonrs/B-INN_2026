pragma solidity ^0.8.19;

interface IRegistration {
    function getVCBinding(bytes32 vcId) external view returns (
        bytes32 vcHash,
        address owner,
        bytes32 imageHash,
        uint256 authThreshold,
        uint256 issuedAt,
        uint256 expiredAt,
        bool active
    );

    function isValidVC(bytes32 vcId) external view returns (bool);
}

contract verification {
    uint256 public constant HASH_BITS = 256;
    uint256 public constant ACCURACY_SCALE = 10000;

    address public contractOwner;
    IRegistration public registrationContract;

    struct OwnershipRecord {
        bytes32 imageHash;
        bytes32 vcId;
        bytes32 vcHash;
        address owner;
        uint256 authThreshold;
        uint256 registeredAt;
    }

    mapping(bytes32 => OwnershipRecord) public ownershipMap;

    event RegistrationContractSet(address indexed registrationContract);
    event OwnershipRegistered(
        bytes32 indexed imageHash,
        bytes32 indexed vcId,
        address indexed owner,
        bytes32 vcHash,
        uint256 authThreshold
    );
    event OwnershipVerified(
        bytes32 indexed imageHash,
        address indexed claimant,
        bool result,
        uint256 bitAccuracy
    );

    modifier onlyContractOwner() {
        require(msg.sender == contractOwner, "Not contract owner");
        _;
    }

    constructor() {
        contractOwner = msg.sender;
    }

    function setRegistrationContract(address registrationAddress) external onlyContractOwner {
        require(registrationAddress != address(0), "Invalid registration address");
        registrationContract = IRegistration(registrationAddress);
        emit RegistrationContractSet(registrationAddress);
    }

    function registerOwnership(bytes32 vcId) external {
        require(address(registrationContract) != address(0), "Registration contract not set");

        (
            bytes32 vcHash,
            address owner,
            bytes32 imageHash,
            uint256 authThreshold,
            ,
            ,
            bool active
        ) = registrationContract.getVCBinding(vcId);

        require(active, "VC is not active");
        require(owner == msg.sender, "Claimant is not VC owner");
        require(ownershipMap[imageHash].owner == address(0), "Image already registered");
        require(authThreshold <= ACCURACY_SCALE, "Invalid threshold");

        ownershipMap[imageHash] = OwnershipRecord({
            imageHash: imageHash,
            vcId: vcId,
            vcHash: vcHash,
            owner: owner,
            authThreshold: authThreshold,
            registeredAt: block.timestamp
        });

        emit OwnershipRegistered(imageHash, vcId, owner, vcHash, authThreshold);
    }

    function verifyOwnership(
        bytes32 imageHash,
        bytes32 extractedHash
    ) external returns (bool) {
        OwnershipRecord storage record = ownershipMap[imageHash];
        require(record.owner != address(0), "No ownership record found");
        require(record.owner == msg.sender, "Claimant is not registered owner");
        require(registrationContract.isValidVC(record.vcId), "VC is not active");

        uint256 bitAccuracy = _bitAccuracyScaled(extractedHash, record.vcHash);
        bool verified = extractedHash == record.vcHash || bitAccuracy >= record.authThreshold;

        emit OwnershipVerified(imageHash, msg.sender, verified, bitAccuracy);
        return verified;
    }

    function checkOwnership(
        bytes32 imageHash,
        bytes32 extractedHash,
        address claimant
    ) external view returns (
        bool verified,
        uint256 bitAccuracy,
        address registeredOwner,
        bytes32 registeredVCHash,
        uint256 authThreshold
    ) {
        OwnershipRecord storage record = ownershipMap[imageHash];
        if (record.owner == address(0)) {
            return (false, 0, address(0), bytes32(0), 0);
        }

        bitAccuracy = _bitAccuracyScaled(extractedHash, record.vcHash);
        bool registryOK = record.owner == claimant && registrationContract.isValidVC(record.vcId);
        bool hashOK = extractedHash == record.vcHash || bitAccuracy >= record.authThreshold;

        return (registryOK && hashOK, bitAccuracy, record.owner, record.vcHash, record.authThreshold);
    }

    function getOwnership(bytes32 imageHash) external view returns (
        address owner,
        bytes32 vcId,
        bytes32 vcHash,
        uint256 authThreshold,
        uint256 registeredAt
    ) {
        OwnershipRecord storage record = ownershipMap[imageHash];
        return (record.owner, record.vcId, record.vcHash, record.authThreshold, record.registeredAt);
    }

    function _bitAccuracyScaled(bytes32 extractedHash, bytes32 registeredHash) internal pure returns (uint256) {
        uint256 mismatches = _hammingDistance(extractedHash, registeredHash);
        return ((HASH_BITS - mismatches) * ACCURACY_SCALE) / HASH_BITS;
    }

    function _hammingDistance(bytes32 a, bytes32 b) internal pure returns (uint256) {
        uint256 x = uint256(a) ^ uint256(b);
        uint256 count = 0;

        while (x != 0) {
            unchecked {
                x &= (x - 1);
                count += 1;
            }
        }

        return count;
    }
}
