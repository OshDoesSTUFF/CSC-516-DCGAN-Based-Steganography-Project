import os
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# ------------------ CONFIG ------------------
decoder_path = "output/netDec.pth"  # path to trained decoder
nz = 100
image_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------ DECODER MODEL ------------------
class Decoder(nn.Module):
    def __init__(self, nz=100):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, nz),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

def noise_to_message(decoded_z, threshold=0.0):
    bits = ['1' if val > threshold else '0' for val in decoded_z.squeeze().tolist()]
    chars = [chr(int(''.join(bits[i:i+8]), 2)) for i in range(0, len(bits), 8)]
    return ''.join(chars).strip('\x00'), bits

def load_decoder(decoder_path=decoder_path, nz=nz):
    netDec = Decoder(nz).to(device)
    netDec.load_state_dict(torch.load(decoder_path, map_location=device))
    netDec.eval()
    return netDec

def decode_image(image_path, decoder_path=decoder_path, nz=nz, image_size=image_size):
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"File '{image_path}' not found.")

    netDec = load_decoder(decoder_path, nz)
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        recovered_z = netDec(image_tensor).cpu()
    decoded_msg, bits = noise_to_message(recovered_z)
    return decoded_msg

if __name__ == "__main__":
    filename = input("Enter the name of the image file in the 'output' folder (e.g., stego_result.png): ")
    image_path = os.path.join("output", filename)
    try:
        decoded_msg = decode_image(image_path)
        print("\nDecoded message from image:")
        print(decoded_msg)
    except Exception as e:
        print(str(e))