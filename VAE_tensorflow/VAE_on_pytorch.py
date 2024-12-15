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

# Параметры
img_height, img_width = 128, 128
batch_size = 32
latent_dim = 200
epochs = 50
learning_rate = 0.0005

# Препроцессинг данных
transform = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor(),
])

# Датасеты
# main_path = r'A:\pycharm_projects\kaggle_smile_no_smile\datasets\from_kaggle\test_predicted'
main_path = r'A:\pycharm_projects\kaggle_smile_no_smile\datasets\from_kaggle\train_and_valid'
train_dataset = datasets.ImageFolder(main_path, transform=transform)
train_size = int(0.9 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# Энкодер
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv_layers = nn.Sequential(
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
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc(x)
        mean = self.mean(x)
        log_variance = self.log_variance(x)
        # log_variance = torch.clamp(log_variance, min=-10, max=10)
        return mean, log_variance

# Декодер
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 4096)
        self.unflatten = nn.Unflatten(1, (64, 8, 8))
        self.deconv_layers = nn.Sequential(
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
        x = self.deconv_layers(x)
        return x

# Вариационный автокодировщик
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def reparameterize(self, mean, log_variance):
        std = torch.exp(0.5 * log_variance)
        epsilon = torch.randn_like(std)
        return mean + epsilon * std

    def forward(self, x):
        mean, log_variance = self.encoder(x)
        z = self.reparameterize(mean, log_variance)
        reconstructed = self.decoder(z)
        return reconstructed, mean, log_variance

# Функция потерь
def vae_loss(reconstructed, original, mean, log_variance):
    reconstruction_loss = nn.MSELoss(reduction='mean')(reconstructed, original)
    kl_divergence = -0.5 * torch.sum(1 + log_variance - mean.pow(2) - log_variance.exp())
    return reconstruction_loss + kl_divergence

# Модель и оптимизатор
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vae = VAE().to(device)
print(summary(vae.encoder, input_size=(3, 128, 128)))
optimizer = optim.Adam(vae.parameters(), lr=learning_rate)

# Обучение
vae.train()
t_losses = []
b_losses = []
iteration = 0
for epoch in range(epochs):
    epoch_loss = 0
    for images, _ in tqdm(train_loader):
        images = images.to(device)
        optimizer.zero_grad()
        reconstructed, mean, log_variance = vae(images)
        loss = vae_loss(reconstructed, images, mean, log_variance)
        b_losses.append(loss.to('cpu').detach().numpy())
        iteration += 1
        if torch.isnan(loss):
            print("NaN detected!")
            print(f"Mean: {mean}")
            print(f"Log Variance: {log_variance}")
            print(f"Reconstructed: {reconstructed}")
            break
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_loader.dataset):.4f}")
    if epoch % 10 == 0:
        print(
            f'Epoch: {epoch} | Batch Loss: {loss} | Iteration: {++iteration} | Running loss {np.average(b_losses[-100:])}')
    t_losses.append(np.average(b_losses[:100]))

# Сохранение модели
torch.save(vae.state_dict(), 'vae.pth')

save_dir=r'A:\pycharm_projects\kaggle_smile_no_smile\Celeba\output\reconstruction'
for i in range(len(images)):
    output_tensor=vae(images[i].unsqueeze(0))
    predicted_img=output_tensor[0].to('cpu').detach().numpy()
    predicted_img = predicted_img.squeeze()
    predicted_img = predicted_img.transpose((1, 2, 0))
    predicted_img = np.clip(predicted_img, a_min=0, a_max=1)
    original_image = images[i]
    original_image = original_image.to('cpu').detach().numpy()
    original_image = original_image.transpose((1, 2, 0))
    original_image = np.clip(original_image, a_min=0, a_max=1)
    combined_img = np.concatenate([original_image, predicted_img], axis=1)
    matplotlib.image.imsave(os.path.join(save_dir,f"{i}.jpg"),combined_img)
