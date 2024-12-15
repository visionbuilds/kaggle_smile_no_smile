import os
import cv2
import numpy as np
# from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1  # Для работы с лицами

# Параметры
image_size = 128
latent_dim = 512
batch_size = 64
epochs = 50
learning_rate = 0.001
validation_split = 0.2
model_save_path = "best_vae_model.pth"


# Подготовка данных
class FaceDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_paths = [
            os.path.join(folder_path, fname)
            for fname in os.listdir(folder_path)
            if fname.endswith(".jpg")
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (image_size, image_size))
        if self.transform:
            image = self.transform(image)
        return image


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Загружаем датасет и делим на обучающую и проверочную выборки
dataset = FaceDataset("A:\pycharm_projects\kaggle_smile_no_smile\datasets\Celeba\cropped", transform=transform)
# dataset = FaceDataset("A:\pycharm_projects\kaggle_smile_no_smile\datasets\Celeba\small", transform=transform)
train_size = int((1 - validation_split) * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# Модель энкодера и декодера
class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.encoder = InceptionResnetV1(pretrained='vggface2').eval()
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(1, (512, 4, 4)),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32,3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def encode(self, x):
        with torch.no_grad():
            features = self.encoder(x).detach()
        mu = self.fc_mu(features)
        logvar = self.fc_logvar(features)
        return mu, logvar

    def decode(self, z):
        x_reconstructed = self.decoder(z)
        return x_reconstructed

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_reconstructed = self.decode(z)
        return x_reconstructed, mu, logvar


# Функция потерь
def vae_loss(reconstructed, original, mu, logvar):
    recon_loss = nn.MSELoss()(reconstructed, original)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_div / original.size(0)


# Инициализация модели и оптимизатора
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = VAE(latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Переменная для хранения минимального loss на валидации
best_val_loss = float("inf")

# Обучение
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for images in train_loader:
        images = images.to(device)
        optimizer.zero_grad()
        reconstructed, mu, logvar = model(images)
        loss = vae_loss(reconstructed, images, mu, logvar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Валидация
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images in val_loader:
            images = images.to(device)
            reconstructed, mu, logvar = model(images)
            loss = vae_loss(reconstructed, images, mu, logvar)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {total_loss / len(train_loader):.4f}, Val Loss: {val_loss:.4f}")

    # Сохранение лучшей модели
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), model_save_path)
        print(f"New best model saved at epoch {epoch + 1}")