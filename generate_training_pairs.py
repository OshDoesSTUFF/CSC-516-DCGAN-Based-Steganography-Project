"""
generate_training_pairs.py
--------------------------
Generate training data for the Learned Latent Denoiser.

Author: Daniel Muskat
Course: CSC 616

Overview:
    This script generates thousands of (clean_z, noisy_z_hat) pairs by:
        1. Sampling random latent vectors z
        2. Running them through the trained Generator to produce images
        3. Applying JPEG compression to those images
        4. Running the compressed images through the trained Decoder
        5. Saving the (z, z_hat) pairs to disk for Corrector training

    The Corrector network will learn to map noisy z_hat → clean z,
    effectively learning the specific error fingerprint that JPEG
    compression produces on THIS model's latent space.

    This is different from general-purpose error correction (Hamming,
    repetition) because it is trained on the actual error distribution
    of this specific Generator/Decoder pair under real JPEG compression.

Usage:
    python generate_training_pairs.py

Output:
    training_pairs/pairs_q75.pt   — 10,000 pairs at JPEG QF=75
    training_pairs/pairs_q30.pt   — 10,000 pairs at JPEG QF=30
    training_pairs/pairs_clean.pt — 5,000 pairs with no compression (baseline)

Requirements:
    - Trained model weights in output/ (netG.pth, netDec.pth)
    - generate_stego_image.py and decode_stego_image.py in same directory

Estimated runtime on M3: 2-4 hours for 25,000 total pairs
"""

import os
import torch
import torch.nn as nn
from torchvision import transforms, utils
from PIL import Image
from io import BytesIO
import time

from generate_stego_image import load_generator
from decode_stego_image import load_decoder

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

PAIRS_PER_CONDITION = 10000   # pairs at each JPEG quality
CLEAN_PAIRS = 5000            # pairs with no compression
BATCH_SIZE = 64               # generate this many images at once (M3 friendly)
OUTPUT_DIR = "training_pairs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NZ = 100

device = torch.device("mps" if torch.backends.mps.is_available()
                       else "cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")
print(f"Generating {PAIRS_PER_CONDITION} pairs at QF=75")
print(f"Generating {PAIRS_PER_CONDITION} pairs at QF=30")
print(f"Generating {CLEAN_PAIRS} pairs with no compression")
print(f"Total pairs: {PAIRS_PER_CONDITION * 2 + CLEAN_PAIRS}")
print(f"Output directory: {OUTPUT_DIR}/")
print()


# ---------------------------------------------------------------------------
# JPEG SIMULATION (inline, no file I/O for speed)
# ---------------------------------------------------------------------------

def jpeg_compress_tensor_batch(img_tensor, quality):
    """
    Apply JPEG compression to a batch of image tensors in memory.
    No disk I/O — uses BytesIO for speed.

    Args:
        img_tensor: (B, C, H, W) tensor, values in [-1, 1]
        quality: JPEG quality factor (1-95)

    Returns:
        Compressed and decompressed tensor, same shape
    """
    imgs = []
    for img in img_tensor:
        img_cpu = img.cpu()
        img_01 = (img_cpu * 0.5 + 0.5).clamp(0, 1)
        img_pil = transforms.ToPILImage()(img_01)
        buffer = BytesIO()
        img_pil.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        img_jpeg = Image.open(buffer).convert('RGB')
        img_back = transforms.ToTensor()(img_jpeg)
        img_normalized = (img_back - 0.5) / 0.5
        imgs.append(img_normalized)
    return torch.stack(imgs).to(img_tensor.device)


# ---------------------------------------------------------------------------
# PAIR GENERATION
# ---------------------------------------------------------------------------

def generate_pairs(n_pairs, jpeg_quality, label):
    """
    Generate n_pairs of (clean_z, noisy_z_hat) training examples.

    Args:
        n_pairs: Number of pairs to generate
        jpeg_quality: JPEG quality factor, or None for no compression
        label: String label for progress reporting

    Returns:
        Tuple of tensors: (clean_z_tensor, noisy_z_hat_tensor)
        Both shape: (n_pairs, NZ)
    """
    netG = load_generator().to(device)
    netDec = load_decoder().to(device)
    netG.eval()
    netDec.eval()

    all_clean_z = []
    all_noisy_z_hat = []

    n_batches = (n_pairs + BATCH_SIZE - 1) // BATCH_SIZE
    start_time = time.time()

    print(f"Generating {n_pairs} pairs ({label})...")

    with torch.no_grad():
        for batch_idx in range(n_batches):
            current_batch = min(BATCH_SIZE, n_pairs - batch_idx * BATCH_SIZE)

            # Step 1: Sample random latent vectors
            z = torch.randn(current_batch, NZ, 1, 1, device=device)
            clean_z_flat = z.view(current_batch, NZ)

            # Step 2: Generate images
            fake_images = netG(z)

            # Step 3: Apply compression (or not)
            if jpeg_quality is not None:
                compressed_images = jpeg_compress_tensor_batch(fake_images, jpeg_quality)
            else:
                compressed_images = fake_images

            # Step 4: Decode to get noisy latent estimate
            noisy_z_hat = netDec(compressed_images)

            # Step 5: Collect pairs
            all_clean_z.append(clean_z_flat.cpu())
            all_noisy_z_hat.append(noisy_z_hat.cpu())

            # Progress reporting
            pairs_done = (batch_idx + 1) * BATCH_SIZE
            pairs_done = min(pairs_done, n_pairs)
            elapsed = time.time() - start_time
            rate = pairs_done / elapsed if elapsed > 0 else 0
            eta = (n_pairs - pairs_done) / rate if rate > 0 else 0

            if (batch_idx + 1) % 10 == 0 or batch_idx == n_batches - 1:
                print(f"  [{pairs_done:>6}/{n_pairs}]  "
                      f"Rate: {rate:.0f} pairs/sec  "
                      f"ETA: {eta/60:.1f} min")

    clean_z_tensor = torch.cat(all_clean_z, dim=0)[:n_pairs]
    noisy_z_hat_tensor = torch.cat(all_noisy_z_hat, dim=0)[:n_pairs]

    elapsed_total = time.time() - start_time
    print(f"  Done. {n_pairs} pairs in {elapsed_total/60:.1f} minutes.\n")

    return clean_z_tensor, noisy_z_hat_tensor


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("Learned Denoiser — Training Data Generation")
    print("=" * 65)
    print()

    total_start = time.time()

    # Generate pairs at QF=75
    clean_75, noisy_75 = generate_pairs(PAIRS_PER_CONDITION, 75, "JPEG QF=75")
    torch.save({
        "clean_z": clean_75,
        "noisy_z_hat": noisy_75,
        "jpeg_quality": 75,
        "n_pairs": PAIRS_PER_CONDITION,
        "nz": NZ
    }, os.path.join(OUTPUT_DIR, "pairs_q75.pt"))
    print(f"Saved: {OUTPUT_DIR}/pairs_q75.pt")

    # Generate pairs at QF=30
    clean_30, noisy_30 = generate_pairs(PAIRS_PER_CONDITION, 30, "JPEG QF=30")
    torch.save({
        "clean_z": clean_30,
        "noisy_z_hat": noisy_30,
        "jpeg_quality": 30,
        "n_pairs": PAIRS_PER_CONDITION,
        "nz": NZ
    }, os.path.join(OUTPUT_DIR, "pairs_q30.pt"))
    print(f"Saved: {OUTPUT_DIR}/pairs_q30.pt")

    # Generate clean pairs (no compression — baseline for comparison)
    clean_nc, noisy_nc = generate_pairs(CLEAN_PAIRS, None, "No compression")
    torch.save({
        "clean_z": clean_nc,
        "noisy_z_hat": noisy_nc,
        "jpeg_quality": None,
        "n_pairs": CLEAN_PAIRS,
        "nz": NZ
    }, os.path.join(OUTPUT_DIR, "pairs_clean.pt"))
    print(f"Saved: {OUTPUT_DIR}/pairs_clean.pt\n")

    # Summary
    total_elapsed = time.time() - total_start
    print("=" * 65)
    print(f"Generation complete.")
    print(f"Total pairs: {PAIRS_PER_CONDITION * 2 + CLEAN_PAIRS:,}")
    print(f"Total time:  {total_elapsed/60:.1f} minutes")
    print(f"Files saved to: {OUTPUT_DIR}/")
    print()
    print("Next step: run train_corrector.py")
    print("=" * 65)


if __name__ == "__main__":
    main()
