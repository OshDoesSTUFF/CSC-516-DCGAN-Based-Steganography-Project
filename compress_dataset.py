import os
from PIL import Image

input_dir = 'img_align_celeba'
output_dir = 'img_align_celeba_compressed'
compression_quality = 50  # Adjust quality (1-95), lower means more compression

os.makedirs(output_dir, exist_ok=True)

for i, filename in enumerate(os.listdir(input_dir), 1):
    if filename.lower().endswith(('.jpg', '.jpeg')):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        with Image.open(input_path) as img:
            img.save(output_path, 'JPEG', quality=compression_quality, optimize=True)
        if i % 1000 == 0:
            print(f'Processed {i} images...')