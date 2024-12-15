import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import os
from torchsummary import summary



# Define the encoder
class Encoder(nn.Module):
    def __init__(self, img_height, img_width, latent_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.25),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.25),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.25),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.25),
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(8 * 8 * 64, 4096)
        self.mean = nn.Linear(4096, latent_dim)
        self.log_variance = nn.Linear(4096, latent_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.fc(x)
        mean = self.mean(x)
        log_variance = self.log_variance(x)
        return mean, log_variance

# Define the decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim, img_height, img_width):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 4096)
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(64, 8, 8))
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.25),
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.25),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.25),
            nn.ConvTranspose2d(32, 3, kernel_size=2, stride=2),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.fc(x)
        x = self.unflatten(x)
        x = self.decoder(x)
        return x

# Sampling function
def sample_latent(mean, log_variance):
    std = torch.exp(0.5 * log_variance)
    epsilon = torch.randn_like(std)
    return mean + epsilon * std

# Loss functions
def reconstruction_loss(y_true, y_pred):
    return nn.MSELoss(reduction='sum')(y_pred, y_true)

def kl_loss(mean, log_variance):
    return -0.5 * torch.sum(1 + log_variance - mean.pow(2) - log_variance.exp())

def vae_loss(y_true, y_pred, mean, log_variance,epoch):
    recon_loss = reconstruction_loss(y_true, y_pred)
    kl = kl_loss(mean, log_variance)
    if epoch % 50 == 0:
        print(f"recon_loss {recon_loss}, kl loss {kl}")
    return recon_loss + kl



# Parameters
img_height, img_width = 128, 128
latent_dim = 200
epochs=1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])
# Initialize the model, optimizer, and dataloader
encoder = Encoder(img_height, img_width, latent_dim).to(device)
decoder = Decoder(latent_dim, img_height, img_width).to(device)
optimizer_enc = optim.Adam(encoder.parameters(), lr=0.0005)
optimizer_dec = optim.Adam(decoder.parameters(), lr=0.0005)

# Replace `your_dataloader` with your dataset DataLoader
main_path = r'A:\pycharm_projects\kaggle_smile_no_smile\datasets\from_kaggle\train_and_valid'
dataset = datasets.ImageFolder(root=main_path, transform=transform)
dataset = [inputs.to(device) for inputs, targets in dataset]
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=2)


encoder.train()
decoder.train()
losses = []
best_encoder=None
best_decoder=None
best_loss=10000000000
best_epoch=0
for epoch in tqdm(range(epochs)):
    batch_losses = []
    for images, _ in dataloader:
        images = images.to(device)

        # Forward pass
        mean, log_variance = encoder(images)
        if torch.isnan(mean).any() or torch.isinf(mean).any():
            print("NaN or Inf detected in mean")
            break

        if torch.isnan(log_variance).any() or torch.isinf(log_variance).any():
            print("NaN or Inf detected in log_variance")
            break
        latent = sample_latent(mean, log_variance)
        reconstructed = decoder(latent)

        # Compute loss
        loss = vae_loss(images, reconstructed, mean, log_variance,epoch)

        # Backward pass
        optimizer_enc.zero_grad()
        optimizer_dec.zero_grad()
        loss.backward()
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"NaN or Inf detected in loss at epoch {epoch}")
            break
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
        optimizer_enc.step()
        optimizer_dec.step()

        batch_losses.append(loss.item())
    epoch_loss = sum(batch_losses) / len(batch_losses)
    losses.append(epoch_loss)
    if epoch_loss < best_loss and epoch > 150 and (epoch - best_epoch) > 20:
        best_epoch = epoch
        best_loss = epoch_loss
        best_encoder = encoder
        best_decoder = decoder
        print(f"epoch {epoch} save best models")
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {epoch_loss}")

torch.save(encoder.state_dict(), "encoder_weights.pth")
torch.save(decoder.state_dict(), "decoder_weights.pth")
torch.save({
    'model_state_dict': encoder.state_dict(),
    'optimizer_state_dict': optimizer_enc.state_dict(),
}, "checkpoint_encoder.pth")
torch.save({
    'model_state_dict': decoder.state_dict(),
    'optimizer_state_dict': optimizer_dec.state_dict(),
}, "checkpoint_decoder.pth")



save_dir=r'A:\pycharm_projects\kaggle_smile_no_smile\Celeba\output\reconstruction'
for i in range(len(images)):

    predicted_img=reconstructed[i].to('cpu').detach().numpy()
    predicted_img = predicted_img.squeeze()
    predicted_img = predicted_img.transpose((1, 2, 0))
    predicted_img = np.clip(predicted_img, a_min=0, a_max=1)

    original_image = images[i]
    original_image = original_image.to('cpu').detach().numpy()
    original_image = original_image.transpose((1, 2, 0))
    original_image = np.clip(original_image, a_min=0, a_max=1)
    combined_img = np.concatenate([original_image, predicted_img], axis=1)
    matplotlib.image.imsave(os.path.join(save_dir,f"{i}.jpg"),combined_img)