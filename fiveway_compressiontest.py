"""
fiveway_compressiontest.py
--------------------------
Five-way comparative test:
    Standard | Hamming | 3x Repetition | Confidence | Learned Denoiser

Author: Daniel Muskat
Course: CSC 616

This is the final and definitive comparison, adding the Learned Latent
Denoiser (Corrector network) to the previous four-way test. The Corrector
is a neural network trained specifically on the error patterns that JPEG
compression produces on this model's latent space. Unlike all previous
approaches, it learns from data rather than applying fixed rules.

Hypothesis:
    A neural network trained on thousands of (noisy_z_hat, clean_z) pairs
    will learn the specific statistical structure of JPEG errors in this
    model's latent space and outperform all hand-crafted error correction
    schemes, including 3x repetition coding.

Usage:
    python fiveway_compressiontest.py

    Run this AFTER train_corrector.py has completed.

Requirements:
    - output/netG.pth, output/netDec.pth  (original trained models)
    - output/corrector.pth                (trained Corrector network)
    - All previous codec files in same directory
"""

import os
import random
import string
from PIL import Image
from io import BytesIO

from generate_stego_image import load_generator
from decode_stego_image import load_decoder
from hamming_codec import message_to_noise_hamming, noise_to_message_hamming
from repetition_codec import message_to_noise_repetition, noise_to_message_repetition
from confidence_decoder import decode_ensemble

import torch
import torch.nn as nn
from torchvision import utils, transforms

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

NUM_TESTS = 50
MSG_LEN = 4
JPEG_QUALITIES = [None, 75, 30]
OUTPUT_DIR = "fiveway_test_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("mps" if torch.backends.mps.is_available()
                       else "cuda" if torch.cuda.is_available() else "cpu")
NZ = 100


# ---------------------------------------------------------------------------
# CORRECTOR NETWORK (must match train_corrector.py architecture)
# ---------------------------------------------------------------------------

class Corrector(nn.Module):
    def __init__(self, nz=NZ, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 512, 512, 256, 128]
        layers = []
        in_dim = nz
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.2, inplace=True),
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, nz))
        self.network = nn.Sequential(*layers)
        self.output_activation = nn.Tanh()

    def forward(self, x):
        delta = self.network(x)
        corrected = x + delta * 0.5
        return self.output_activation(corrected)


def load_corrector():
    corrector_path = os.path.join("output", "corrector.pth")
    if not os.path.exists(corrector_path):
        raise FileNotFoundError(
            f"Corrector not found at {corrector_path}. "
            "Run train_corrector.py first."
        )
    checkpoint = torch.load(corrector_path, map_location=device)
    hidden_dims = checkpoint.get("hidden_dims", [256, 512, 512, 256, 128])
    model = Corrector(nz=NZ, hidden_dims=hidden_dims).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"  Loaded Corrector from epoch {checkpoint['epoch']} "
          f"(val bit acc: {checkpoint['val_bit_acc']:.2f}%)")
    return model


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def str_to_bits(s):
    return ''.join(f'{ord(c):08b}' for c in s)


def bit_match_percent(original, recovered):
    orig_bits = str_to_bits(original)
    rec_bits = str_to_bits(recovered[:len(original)])
    min_len = min(len(orig_bits), len(rec_bits))
    if min_len == 0:
        return 0.0
    matches = sum(a == b for a, b in zip(orig_bits[:min_len], rec_bits[:min_len]))
    return matches / len(orig_bits) * 100


def apply_jpeg_bytes(img_tensor, quality):
    """Apply JPEG compression in memory (no disk I/O)."""
    imgs = []
    for img in img_tensor:
        img_01 = (img.cpu() * 0.5 + 0.5).clamp(0, 1)
        img_pil = transforms.ToPILImage()(img_01)
        buf = BytesIO()
        img_pil.save(buf, format='JPEG', quality=quality)
        buf.seek(0)
        img_back = transforms.ToTensor()(Image.open(buf).convert('RGB'))
        imgs.append((img_back - 0.5) / 0.5)
    return torch.stack(imgs)


def get_raw_latent(latent_floats, prefix, idx, jpeg_quality):
    """
    Generate image, compress, decode — return raw Decoder float output.
    All schemes share this image generation step for fair comparison.
    """
    netG = load_generator().to(device)
    netDec = load_decoder().to(device)

    z = torch.tensor(latent_floats, dtype=torch.float32).view(1, NZ, 1, 1).to(device)

    with torch.no_grad():
        img = netG(z)
        if jpeg_quality is not None:
            img_compressed = apply_jpeg_bytes(img.cpu(), jpeg_quality).to(device)
        else:
            img_compressed = img
        raw_z = netDec(img_compressed).cpu().squeeze().tolist()

    return raw_z


def standard_encode(message):
    bits = []
    for char in message:
        bits.extend([int(b) for b in format(ord(char), '08b')])
    bits = bits[:NZ]
    bits += [0] * (NZ - len(bits))
    return [1.0 if b == 1 else -1.0 for b in bits]


def standard_decode_n(latent_values, n_chars):
    bits = ['1' if v > 0.0 else '0' for v in latent_values[:n_chars*8]]
    chars = [chr(int(''.join(bits[i:i+8]), 2)) for i in range(0, n_chars*8, 8)]
    return ''.join(chars).replace('\x00', '')


def corrector_decode(latent_values, corrector, n_chars):
    """Apply Corrector network then decode."""
    z_hat = torch.tensor(latent_values, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        corrected = corrector(z_hat).cpu().squeeze().tolist()
    bits = ['1' if v > 0.0 else '0' for v in corrected[:n_chars*8]]
    chars = [chr(int(''.join(bits[i:i+8]), 2)) for i in range(0, n_chars*8, 8)]
    return ''.join(chars).replace('\x00', '')


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 75)
    print("Five-Way Encoding Comparison")
    print("Standard | Hamming | Repetition | Confidence | Learned Denoiser")
    print(f"Messages: {NUM_TESTS} random {MSG_LEN}-char strings  |  Device: {device}")
    print("=" * 75)

    # Load corrector
    print("\nLoading Corrector network...")
    corrector = load_corrector()

    # Shared messages
    random.seed(42)
    messages = [''.join(random.choices(string.ascii_lowercase, k=MSG_LEN))
                for _ in range(NUM_TESTS)]

    condition_labels = {None: "No compression (PNG)", 75: "JPEG QF=75", 30: "JPEG QF=30"}

    scheme_names = [
        "Standard",
        "Hamming(7,4)",
        "3x Repetition",
        "Confidence",
        "Learned Denoiser",
    ]

    all_results = {}

    for quality in JPEG_QUALITIES:
        label = condition_labels[quality]
        print(f"\n--- {label} ---")

        res = {s: {"exact": 0, "bits": []} for s in scheme_names}

        for i, msg in enumerate(messages):
            prefix = f"q{quality}" if quality else "clean"

            # Standard encoding — get raw latent for reuse
            std_floats = standard_encode(msg)
            raw_z = get_raw_latent(std_floats, f"std_{prefix}", i, quality)

            # Standard decode
            std_dec = standard_decode_n(raw_z, MSG_LEN)
            if std_dec == msg:
                res["Standard"]["exact"] += 1
            res["Standard"]["bits"].append(bit_match_percent(msg, std_dec))

            # Confidence (Ensemble) — same raw_z, different decode
            conf_dec = decode_ensemble(raw_z, MSG_LEN).replace('\x00', '')
            if conf_dec == msg:
                res["Confidence"]["exact"] += 1
            res["Confidence"]["bits"].append(bit_match_percent(msg, conf_dec))

            # Learned Denoiser — same raw_z, run through Corrector
            corr_dec = corrector_decode(raw_z, corrector, MSG_LEN)
            if corr_dec == msg:
                res["Learned Denoiser"]["exact"] += 1
            res["Learned Denoiser"]["bits"].append(bit_match_percent(msg, corr_dec))

            # Hamming — different encoding, separate image
            _, ham_floats = message_to_noise_hamming(msg)
            ham_z = get_raw_latent(ham_floats, f"ham_{prefix}", i, quality)
            ham_dec = noise_to_message_hamming(ham_z).replace('\x00', '')
            if ham_dec == msg:
                res["Hamming(7,4)"]["exact"] += 1
            res["Hamming(7,4)"]["bits"].append(bit_match_percent(msg, ham_dec))

            # Repetition — different encoding, separate image
            _, rep_floats = message_to_noise_repetition(msg)
            rep_z = get_raw_latent(rep_floats, f"rep_{prefix}", i, quality)
            rep_dec = noise_to_message_repetition(rep_z).replace('\x00', '')
            if rep_dec == msg:
                res["3x Repetition"]["exact"] += 1
            res["3x Repetition"]["bits"].append(bit_match_percent(msg, rep_dec))

        for s in scheme_names:
            res[s]["avg_bit_acc"] = sum(res[s]["bits"]) / NUM_TESTS
            res[s]["exact_pct"] = res[s]["exact"] / NUM_TESTS * 100
            name = s
            print(f"  {name:<20} bit acc: {res[s]['avg_bit_acc']:.2f}%  "
                  f"exact: {res[s]['exact_pct']:.1f}%")

        all_results[label] = res

    # Summary tables
    print("\n\n" + "=" * 75)
    print("FINAL RESULTS — BIT ACCURACY")
    print("=" * 75)
    header = f"{'Condition':<24}" + "".join(f"{s[:14]:>14}" for s in scheme_names)
    print(header)
    print("-" * 75)
    for label, res in all_results.items():
        best = max(res[s]["avg_bit_acc"] for s in scheme_names)
        row = f"{label:<24}"
        for s in scheme_names:
            acc = res[s]["avg_bit_acc"]
            marker = "*" if acc == best else " "
            row += f"{acc:.2f}%{marker:>6}"
        print(row)

    print("\n\nFINAL RESULTS — EXACT MATCH RATE")
    print("=" * 75)
    print(header)
    print("-" * 75)
    for label, res in all_results.items():
        best = max(res[s]["exact_pct"] for s in scheme_names)
        row = f"{label:<24}"
        for s in scheme_names:
            ep = res[s]["exact_pct"]
            marker = "*" if ep == best else " "
            row += f"{ep:.1f}%{marker:>7}"
        print(row)

    print("\n\nCAPACITY REFERENCE")
    print("-" * 75)
    print("  Standard / Confidence / Learned Denoiser : ~12 ASCII chars")
    print("  Hamming(7,4)                              : ~7  ASCII chars")
    print("  3x Repetition                             : ~4  ASCII chars")
    print("\n  * = best performer for that condition")
    print("=" * 75)


if __name__ == "__main__":
    main()
