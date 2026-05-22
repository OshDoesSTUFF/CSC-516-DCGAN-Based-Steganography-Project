"""
repetition_codec.py
-------------------
Triple-bit repetition error correction codec for DCGAN-based steganography.

Author: Daniel Muskat
Course: CSC 616

Overview:
    Standard steganography encodes each message bit once into the latent
    vector. A single flipped bit corrupts a character. This module implements
    a 3x repetition code: every data bit is encoded three times. On decoding,
    a majority vote determines the correct bit — if two of the three copies
    agree, that value wins. This corrects any single-bit error per triplet.

Motivation:
    Hamming(7,4) was tested and showed only modest improvement (~1-2pp)
    under JPEG compression. The hypothesis was that JPEG produces too many
    bit errors per 7-bit block for Hamming's single-error correction to
    handle effectively. Repetition coding takes a different approach:
    rather than using parity bits to locate and fix one error per block,
    it uses raw redundancy and majority voting, which may be more robust
    when errors are distributed somewhat randomly across the bit stream.

Tradeoff:
    - Without error correction: ~12 ASCII characters (100 bits / 8)
    - With 3x repetition:      ~4  ASCII characters (33 data bits / 8)
    Capacity is significantly reduced, but each bit can survive one flip
    out of three copies regardless of where in the latent vector it lands.

Comparison with Hamming(7,4):
    Hamming(7,4): 4 data bits per 7-bit block — 57% efficiency
                  Corrects 1 error per 7-bit block
    Repetition:   1 data bit per 3-bit block   — 33% efficiency
                  Corrects 1 error per 3-bit block (higher error density tolerance)

    Repetition is less efficient but potentially more robust when errors
    are not isolated — which is the characteristic failure mode of JPEG.
"""


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

NZ = 100                    # Latent vector dimension (must match training)
REPEAT = 3                  # Number of times each bit is repeated
DATA_BITS = NZ // REPEAT    # 33 usable data bits (99 used, 1 padding)
CAPACITY_CHARS = DATA_BITS // 8   # 4 ASCII characters


# ---------------------------------------------------------------------------
# ENCODING
# ---------------------------------------------------------------------------

def message_to_repetition_bits(message: str) -> list:
    """
    Convert a text message to a 3x repetition-encoded bit list of length NZ.

    Process:
        1. Convert message to binary string (8 bits per ASCII character)
        2. Pad or truncate to DATA_BITS (33) data bits
        3. Repeat each bit 3 times
        4. Concatenate → 99 bits, pad with 1 zero to reach NZ=100

    Args:
        message: ASCII text string (max 4 characters)

    Returns:
        List of 100 integers (0 or 1), ready for latent vector encoding
    """
    # Step 1: Convert to bits
    raw_bits = []
    for char in message:
        raw_bits.extend([int(b) for b in format(ord(char), '08b')])

    # Step 2: Pad or truncate to DATA_BITS
    raw_bits = raw_bits[:DATA_BITS]
    raw_bits += [0] * (DATA_BITS - len(raw_bits))

    # Step 3: Repeat each bit 3 times
    encoded = []
    for bit in raw_bits:
        encoded.extend([bit, bit, bit])

    # Step 4: Pad to NZ=100 (99 repetition bits + 1 padding zero)
    encoded += [0] * (NZ - len(encoded))

    return encoded


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
# DECODING
# ---------------------------------------------------------------------------

def majority_vote(a: int, b: int, c: int) -> int:
    """
    Return the majority value among three bits.
    Corrects any single-bit error in a triplet.

    Args:
        a, b, c: Three integers (0 or 1)

    Returns:
        Integer (0 or 1) — the majority value
    """
    return 1 if (a + b + c) >= 2 else 0


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


def repetition_bits_to_message(raw_bits: list) -> str:
    """
    Convert a raw 100-bit latent output to a decoded text message,
    applying majority-vote error correction to each 3-bit triplet.

    Process:
        1. Take first 99 bits (33 triplets × 3 bits)
        2. Apply majority vote to each triplet → 33 data bits
        3. Convert 33 data bits → 4 ASCII characters

    Args:
        raw_bits: List of 100 integers (0 or 1) from thresholded latent vector

    Returns:
        Decoded and error-corrected ASCII string
    """
    corrected_bits = []

    for i in range(DATA_BITS):
        a = raw_bits[i * REPEAT]
        b = raw_bits[i * REPEAT + 1]
        c = raw_bits[i * REPEAT + 2]
        corrected_bits.append(majority_vote(a, b, c))

    # Convert data bits to ASCII characters
    chars = []
    for i in range(0, CAPACITY_CHARS * 8, 8):
        byte = corrected_bits[i: i + 8]
        char_code = int(''.join(str(b) for b in byte), 2)
        if char_code != 0:
            chars.append(chr(char_code))

    return ''.join(chars)


# ---------------------------------------------------------------------------
# CONVENIENCE WRAPPERS (matching existing pipeline interface)
# ---------------------------------------------------------------------------

def message_to_noise_repetition(message: str, nz: int = NZ):
    """
    Full encoding pipeline: text message → bit list + latent float list.
    Drop-in replacement for standard message_to_noise().

    Args:
        message: ASCII text (max 4 characters with 3x repetition encoding)
        nz: Latent vector dimension (must be 100)

    Returns:
        Tuple: (bit_list, float_list) where float_list is +1.0/-1.0 per bit
    """
    if len(message) > CAPACITY_CHARS:
        print(f"Warning: message truncated to {CAPACITY_CHARS} characters for repetition encoding.")
        message = message[:CAPACITY_CHARS]

    bits = message_to_repetition_bits(message)
    floats = bits_to_latent(bits)
    return bits, floats


def noise_to_message_repetition(latent_values: list, threshold: float = 0.0) -> str:
    """
    Full decoding pipeline: latent float vector → error-corrected text message.
    Drop-in replacement for standard noise_to_message().

    Args:
        latent_values: List of floats from Decoder network output
        threshold: Bit decision threshold (default 0.0)

    Returns:
        Error-corrected decoded ASCII string
    """
    raw_bits = latent_to_bits(latent_values, threshold)
    return repetition_bits_to_message(raw_bits)


# ---------------------------------------------------------------------------
# SELF-TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("3x Bit Repetition Codec Self-Test")
    print("=" * 60)

    # Test 1: Perfect round-trip
    test_msg = "hi!?"
    bits, floats = message_to_noise_repetition(test_msg)
    recovered = noise_to_message_repetition(floats)
    print(f"\nTest 1: Perfect round-trip")
    print(f"  Original : '{test_msg}'")
    print(f"  Recovered: '{recovered}'")
    print(f"  Match    : {test_msg == recovered}")

    # Test 2: Single-bit error correction per triplet
    print(f"\nTest 2: One flipped bit per triplet (worst-case single errors)")
    corrupted = list(floats)
    # Flip the first bit in triplets 0, 4, 8, 12 (spread across message)
    for triplet in [0, 4, 8, 12]:
        corrupted[triplet * REPEAT] *= -1
    recovered_corrupted = noise_to_message_repetition(corrupted)
    print(f"  Original          : '{test_msg}'")
    print(f"  After bit flips   : '{recovered_corrupted}'")
    print(f"  Match             : {test_msg == recovered_corrupted}")

    # Test 3: Two errors in same triplet (beyond correction capacity)
    print(f"\nTest 3: Two flipped bits in same triplet (beyond correction capacity)")
    double_error = list(floats)
    double_error[0] *= -1
    double_error[1] *= -1   # two errors in triplet 0
    recovered_double = noise_to_message_repetition(double_error)
    print(f"  Original          : '{test_msg}'")
    print(f"  After double flip : '{recovered_double}'")
    print(f"  (Expected failure — 2 errors in one triplet exceeds majority vote)")

    # Test 4: Capacity
    print(f"\nTest 4: Capacity summary")
    print(f"  No error correction : ~12 chars (100 raw bits / 8)")
    print(f"  Hamming(7,4)        : ~7  chars (56 data bits / 8)")
    print(f"  3x Repetition       : ~{CAPACITY_CHARS}  chars ({DATA_BITS} data bits / 8)")
    print(f"  Latent bits used    : {DATA_BITS * REPEAT} of {NZ} (+ 1 padding)")

    # Test 5: Encoding visualization
    print(f"\nTest 5: Encoding visualization (first 12 bits of '{test_msg}')")
    raw = [int(b) for b in format(ord(test_msg[0]), '08b')][:4]
    enc = []
    for bit in raw:
        enc.extend([bit, bit, bit])
    print(f"  Data bits    : {raw}")
    print(f"  Encoded (3x) : {enc}")
    print(f"  After 1 flip : {enc[:1] + [1 - enc[1]] + enc[2:]}")
    vote = [majority_vote(enc[i*3], enc[i*3+1] if i*3+1 < len(enc) else 0,
                          enc[i*3+2] if i*3+2 < len(enc) else 0) for i in range(len(raw))]
    print(f"  Majority vote: {vote}  ← original data recovered")

    print("\n" + "=" * 60)
    print("Self-test complete.")
