"""
hamming_codec.py
----------------
Hamming(7,4) error correction codec for DCGAN-based steganography.

Author: Daniel Muskat
Course: CSC 616

Overview:
    Standard steganography encodes message bits directly into the latent
    vector with no redundancy. A single flipped bit corrupts a character.
    This module adds Hamming(7,4) error correction: every 4 data bits are
    encoded as 7 bits (4 data + 3 parity). On decoding, any single-bit
    error in a 7-bit block is automatically detected and corrected.

Tradeoff:
    - Without Hamming: ~12 ASCII characters (100 bits / 8 bits per char)
    - With Hamming:    ~7 ASCII characters  (56 data bits / 8 bits per char)
    Capacity is reduced but message recovery is substantially more reliable
    under lossy compression.

Hamming(7,4) structure:
    Codeword positions: [p1, p2, d1, p3, d2, d3, d4]
    Parity bit coverage:
        p1 (pos 1): covers positions 1, 3, 5, 7  → p1, d1, d2, d4
        p2 (pos 2): covers positions 2, 3, 6, 7  → p2, d1, d3, d4
        p3 (pos 4): covers positions 4, 5, 6, 7  → p3, d2, d3, d4
"""


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

NZ = 100          # Latent vector dimension (must match training config)
BLOCK_SIZE = 7    # Bits per Hamming codeword
DATA_BITS = 4     # Data bits per codeword
NUM_BLOCKS = NZ // BLOCK_SIZE          # 14 complete blocks
HAMMING_CAPACITY_BITS = NUM_BLOCKS * DATA_BITS  # 56 data bits
HAMMING_CAPACITY_CHARS = HAMMING_CAPACITY_BITS // 8  # 7 ASCII characters


# ---------------------------------------------------------------------------
# HAMMING ENCODING
# ---------------------------------------------------------------------------

def encode_block(d: list) -> list:
    """
    Encode 4 data bits into a 7-bit Hamming codeword.

    Args:
        d: List of 4 integers (0 or 1), the data bits [d1, d2, d3, d4]

    Returns:
        List of 7 integers (0 or 1): [p1, p2, d1, p3, d2, d3, d4]
    """
    d1, d2, d3, d4 = d
    p1 = d1 ^ d2 ^ d4          # parity over positions 1,3,5,7
    p2 = d1 ^ d3 ^ d4          # parity over positions 2,3,6,7
    p3 = d2 ^ d3 ^ d4          # parity over positions 4,5,6,7
    return [p1, p2, d1, p3, d2, d3, d4]


def message_to_hamming_bits(message: str) -> list:
    """
    Convert a text message to a Hamming-encoded bit list of length NZ (100).

    Process:
        1. Convert message to binary string (8 bits per ASCII character)
        2. Pad or truncate to HAMMING_CAPACITY_BITS (56) data bits
        3. Split into 14 groups of 4 data bits
        4. Encode each group as a 7-bit Hamming codeword
        5. Concatenate all 14 codewords → 98 bits
        6. Pad with 2 zeros to reach NZ=100

    Args:
        message: ASCII text string (max 7 characters)

    Returns:
        List of 100 integers (0 or 1), ready for latent vector encoding
    """
    # Step 1: Convert message to bits
    raw_bits = []
    for char in message:
        raw_bits.extend([int(b) for b in format(ord(char), '08b')])

    # Step 2: Pad or truncate to exactly HAMMING_CAPACITY_BITS data bits
    raw_bits = raw_bits[:HAMMING_CAPACITY_BITS]
    raw_bits += [0] * (HAMMING_CAPACITY_BITS - len(raw_bits))

    # Step 3 & 4: Encode each 4-bit block as a 7-bit Hamming codeword
    encoded_bits = []
    for i in range(NUM_BLOCKS):
        block = raw_bits[i * DATA_BITS: (i + 1) * DATA_BITS]
        encoded_bits.extend(encode_block(block))

    # Step 5: Pad to NZ=100 (98 Hamming bits + 2 padding zeros)
    encoded_bits += [0] * (NZ - len(encoded_bits))

    return encoded_bits


def bits_to_latent(bits: list) -> list:
    """
    Convert a bit list (0/1) to a latent vector (+1.0/-1.0).

    Args:
        bits: List of 100 integers (0 or 1)

    Returns:
        List of 100 floats (+1.0 or -1.0)
    """
    return [1.0 if b == 1 else -1.0 for b in bits]


# ---------------------------------------------------------------------------
# HAMMING DECODING
# ---------------------------------------------------------------------------

def decode_block(codeword: list) -> list:
    """
    Decode a 7-bit Hamming codeword, correcting single-bit errors.

    Process:
        1. Compute syndrome bits s1, s2, s3
        2. If syndrome is non-zero, it identifies the error position (1-7)
        3. Flip the bit at that position to correct the error
        4. Extract and return the 4 data bits

    Args:
        codeword: List of 7 integers (0 or 1): [p1, p2, d1, p3, d2, d3, d4]

    Returns:
        List of 4 integers (0 or 1): the corrected data bits [d1, d2, d3, d4]
    """
    cw = list(codeword)  # copy to avoid mutating input

    # Compute syndrome
    s1 = cw[0] ^ cw[2] ^ cw[4] ^ cw[6]  # positions 1,3,5,7
    s2 = cw[1] ^ cw[2] ^ cw[5] ^ cw[6]  # positions 2,3,6,7
    s3 = cw[3] ^ cw[4] ^ cw[5] ^ cw[6]  # positions 4,5,6,7

    # Syndrome value identifies the error position (1-indexed)
    error_pos = s1 * 1 + s2 * 2 + s3 * 4

    if error_pos != 0:
        # Flip the erroneous bit (convert to 0-indexed)
        cw[error_pos - 1] ^= 1

    # Extract data bits from corrected codeword positions 3,5,6,7 (0-indexed: 2,4,5,6)
    return [cw[2], cw[4], cw[5], cw[6]]


def latent_to_bits(latent_values: list, threshold: float = 0.0) -> list:
    """
    Convert a latent vector (floats) back to a bit list via thresholding.

    Args:
        latent_values: List of floats (output from Decoder network)
        threshold: Decision boundary (default 0.0)

    Returns:
        List of integers (0 or 1)
    """
    return [1 if v > threshold else 0 for v in latent_values]


def hamming_bits_to_message(raw_bits: list) -> str:
    """
    Convert a raw 100-bit latent output to a decoded text message,
    applying Hamming error correction to each 7-bit block.

    Process:
        1. Take first 98 bits (14 codewords × 7 bits)
        2. Decode each 7-bit block with single-bit error correction
        3. Reconstruct 56 data bits → 7 ASCII characters

    Args:
        raw_bits: List of 100 integers (0 or 1) from thresholded latent vector

    Returns:
        Decoded and error-corrected ASCII string
    """
    corrected_data_bits = []

    for i in range(NUM_BLOCKS):
        codeword = raw_bits[i * BLOCK_SIZE: (i + 1) * BLOCK_SIZE]
        corrected_data_bits.extend(decode_block(codeword))

    # Convert 56 data bits back to ASCII characters
    chars = []
    for i in range(0, HAMMING_CAPACITY_BITS, 8):
        byte = corrected_data_bits[i: i + 8]
        char_code = int(''.join(str(b) for b in byte), 2)
        if char_code != 0:  # skip null padding
            chars.append(chr(char_code))

    return ''.join(chars)


# ---------------------------------------------------------------------------
# CONVENIENCE WRAPPERS (matching existing pipeline interface)
# ---------------------------------------------------------------------------

def message_to_noise_hamming(message: str, nz: int = NZ):
    """
    Full encoding pipeline: text message → latent float vector + original bits.
    Drop-in replacement for the standard message_to_noise() function.

    Args:
        message: ASCII text (max 7 characters with Hamming encoding)
        nz: Latent vector dimension (must be 100)

    Returns:
        Tuple: (bit_list, float_list) where float_list is +1.0/-1.0 per bit
    """
    if len(message) > HAMMING_CAPACITY_CHARS:
        print(f"Warning: message truncated to {HAMMING_CAPACITY_CHARS} characters for Hamming encoding.")
        message = message[:HAMMING_CAPACITY_CHARS]

    bits = message_to_hamming_bits(message)
    floats = bits_to_latent(bits)
    return bits, floats


def noise_to_message_hamming(latent_values: list, threshold: float = 0.0) -> str:
    """
    Full decoding pipeline: latent float vector → error-corrected text message.
    Drop-in replacement for the standard noise_to_message() function.

    Args:
        latent_values: List of floats from Decoder network output
        threshold: Bit decision threshold (default 0.0)

    Returns:
        Error-corrected decoded ASCII string
    """
    raw_bits = latent_to_bits(latent_values, threshold)
    return hamming_bits_to_message(raw_bits)


# ---------------------------------------------------------------------------
# SELF-TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Hamming(7,4) Codec Self-Test")
    print("=" * 60)

    # Test 1: Perfect round-trip (no errors)
    test_msg = "hello!"
    bits, floats = message_to_noise_hamming(test_msg)
    recovered = noise_to_message_hamming(floats)
    print(f"\nTest 1: Perfect round-trip")
    print(f"  Original : '{test_msg}'")
    print(f"  Recovered: '{recovered}'")
    print(f"  Match    : {test_msg == recovered}")

    # Test 2: Single-bit error correction
    print(f"\nTest 2: Single-bit error injection")
    corrupted_floats = list(floats)
    corrupted_floats[5] *= -1  # flip bit at position 5
    corrupted_floats[22] *= -1  # flip bit at position 22
    corrupted_floats[63] *= -1  # flip bit at position 63
    recovered_corrupted = noise_to_message_hamming(corrupted_floats)
    print(f"  Original          : '{test_msg}'")
    print(f"  After 3 bit flips : '{recovered_corrupted}'")
    print(f"  Match             : {test_msg == recovered_corrupted}")
    print(f"  (Each flip is in a different 7-bit block, so all are corrected)")

    # Test 3: Capacity demonstration
    print(f"\nTest 3: Capacity")
    print(f"  Without Hamming: ~12 ASCII chars (100 raw bits / 8)")
    print(f"  With Hamming:    ~{HAMMING_CAPACITY_CHARS} ASCII chars ({HAMMING_CAPACITY_BITS} data bits / 8)")
    print(f"  Latent bits used: {NUM_BLOCKS * BLOCK_SIZE} of {NZ} (+ 2 padding)")

    # Test 4: Bit accuracy threshold
    print(f"\nTest 4: Partial error simulation (50% bit error rate)")
    import random
    noisy_floats = [f * (-1 if random.random() < 0.20 else 1) for f in floats]
    recovered_noisy = noise_to_message_hamming(noisy_floats)
    print(f"  Original : '{test_msg}'")
    print(f"  Recovered: '{recovered_noisy}' (20% bit error rate)")

    print("\n" + "=" * 60)
    print("Self-test complete.")
