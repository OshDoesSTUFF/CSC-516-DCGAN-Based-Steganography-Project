import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

# ------------------ CONFIG ------------------
data_path = "img_align_celeba_compressed"  # make sure this is the full path with the CelebA Dataset "https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html"
save_dir = "output"
os.makedirs(save_dir, exist_ok=True)

image_size = 64
batch_size = 128
nz = 100  # size of the latent vector
num_epochs = 30
lr = 0.0002
beta1 = 0.5
lambda_recon = 50.0  # weight for reconstruction loss

device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
print("Using device:", device)

# ------------------ DATASET ------------------
class CelebADataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = sorted(glob.glob(os.path.join(root_dir, '*.jpg')))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

# ------------------ MODEL SETUP ------------------
def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False), nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

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

# ------------------ TRAINING ------------------
dataset = CelebADataset(root_dir=data_path, transform=transform)
print("Found", len(dataset), "images")

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

netG = Generator().to(device)
netD = Discriminator().to(device)
netDec = Decoder(nz).to(device)
netG.apply(weights_init)
netD.apply(weights_init)
netDec.apply(weights_init)

criterion = nn.BCELoss()
recon_loss = nn.MSELoss()
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerDec = optim.Adam(netDec.parameters(), lr=lr, betas=(beta1, 0.999))

# PLEASE NOTE: if your training the model, leave this as true, else false
training = True

if training:
    for epoch in range(num_epochs):
        for i, real_images in enumerate(tqdm(dataloader), 0):
            real_images = real_images.to(device)
            b_size = real_images.size(0)

            # Train Discriminator
            netD.zero_grad()
            label = torch.full((b_size,), 1.0, device=device)
            output = netD(real_images).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(0.0)
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            optimizerD.step()
            D_G_z1 = output.mean().item()

            # Train Generator + Decoder
            netG.zero_grad()
            netDec.zero_grad()
            label.fill_(1.0)
            output = netD(fake).view(-1)
            errG = criterion(output, label)
            recon = netDec(fake)
            errRecon = recon_loss(recon, noise.view(b_size, -1))
            total_loss = errG + lambda_recon * errRecon
            total_loss.backward()
            optimizerG.step()
            optimizerDec.step()
            D_G_z2 = output.mean().item()

            if i % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] [{i}/{len(dataloader)}]  "
                      f"Loss_D: {(errD_real + errD_fake):.4f}  Loss_G: {errG:.4f}  "
                      f"Recon: {errRecon:.4f}  D(x): {D_x:.4f}  D(G(z)): {D_G_z1:.4f}/{D_G_z2:.4f}")

        with torch.no_grad():
            fake = netG(fixed_noise).detach().cpu()
        utils.save_image(fake, f"{save_dir}/epoch_{epoch+1:03d}.png", normalize=True)

    torch.save(netG.state_dict(), os.path.join(save_dir, "netG.pth"))
    torch.save(netD.state_dict(), os.path.join(save_dir, "netD.pth"))
    torch.save(netDec.state_dict(), os.path.join(save_dir, "netDec.pth"))
    print("\nTraining complete. Models saved.")

# ------------------ TEST DECODING ------------------
def message_to_noise(msg, nz=100):
    bits = ''.join(format(ord(c), '08b') for c in msg)
    bits = bits[:nz] + '0' * (nz - len(bits))  # pad or trim to fit nz
    z = torch.tensor([1.0 if b == '1' else -1.0 for b in bits], dtype=torch.float32)
    return z.view(1, nz, 1, 1).to(device), bits

def noise_to_message(decoded_z, threshold=0.0):
    bits = ['1' if val > threshold else '0' for val in decoded_z.squeeze().tolist()]
    chars = [chr(int(''.join(bits[i:i+8]), 2)) for i in range(0, len(bits), 8)]
    return ''.join(chars).strip('\x00'), bits

# Load trained models
netG.load_state_dict(torch.load(os.path.join(save_dir, "netG.pth"), map_location=device))
netDec.load_state_dict(torch.load(os.path.join(save_dir, "netDec.pth"), map_location=device))
netG.eval()
netDec.eval()

# Encode and decode a test message
test_msg = "hello world"
z_msg, original_bits = message_to_noise(test_msg, nz)

with torch.no_grad():
    stego_img = netG(z_msg)
    utils.save_image(stego_img, os.path.join(save_dir, "stego_result.png"), normalize=True)
    recovered_z = netDec(stego_img).detach().cpu()

decoded_msg, decoded_bits = noise_to_message(recovered_z)

# this measures how closely the reconstructed latent vector (from the image using the decoder) matches the original latent vector (used to generate the image with the generator)
bit_accuracy = sum(o == d for o, d in zip(original_bits, decoded_bits)) / len(original_bits) * 100

print("\nOriginal Message:", test_msg)
print("Decoded Message:", decoded_msg)
print(f"Bit Accuracy: {bit_accuracy:.2f}%")
