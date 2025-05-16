import os
import random
import string
from generate_stego_image import generate_stego_image
from decode_stego_image import decode_image

success_count = 0
results = []

num_tests = 50  # or any number of repetitions

for i in range(num_tests):
    # Generate a random 12-character message (lowercase letters)
    original_msg = ''.join(random.choices(string.ascii_lowercase, k=12))
    filename = f"stego_{i:03d}.png"
    # Generate image
    image_path = generate_stego_image(original_msg, file_name=filename, output_path="images_generated")

    # Decode message directly from the generated PNG (no compression)
    recovered_msg = decode_image(image_path)
    # Compare
    match = original_msg == recovered_msg
    if match:
        success_count += 1

    # Compare at the bit level
    def str_to_bits(s):
        return ''.join(f'{ord(c):08b}' for c in s)

    orig_bits = str_to_bits(original_msg)
    rec_bits = str_to_bits(recovered_msg)
    min_len = min(len(orig_bits), len(rec_bits))
    bit_matches = sum(a == b for a, b in zip(orig_bits[:min_len], rec_bits[:min_len]))
    percent = (bit_matches / len(orig_bits) * 100) if orig_bits else 0
    results.append((original_msg, recovered_msg, percent))

# Print results
for idx, (orig, rec, pct) in enumerate(results):
    print(f"[{idx}] Original:   {orig}")
    print(f"    Recovered: {rec}")
    print(f"    % Match:   {pct:.2f}%\n")

average_percent = sum(pct for _, _, pct in results) / num_tests
print(f"Total exact matches: {success_count}/{num_tests}")
print(f"Overall accuracy: {success_count/num_tests*100:.2f}%")
print(f"Average percent match: {average_percent:.2f}%")