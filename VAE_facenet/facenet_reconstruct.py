import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from PIL import Image
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1  # Для работы с лицами
image_size=128
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


DEVICE = ( "cuda" if torch.cuda.is_available() else "cpu")
model_path = r"A:\pycharm_projects\kaggle_smile_no_smile\VAE\best_vae_model.pth"
latent_dim=128
model = VAE(latent_dim).to(DEVICE)
model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model.eval()
val_transforms = transforms.Compose([
    # transforms.ToPILImage(),
    transforms.Resize((image_size,image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# img_path=r"F:\pycharm_projects_personal\smile_not_smile\VAE\data\Celeba\kaggle\train\Albrecht_Mentz_0001.jpg"
img_path=r"A:\pycharm_projects\kaggle_smile_no_smile\datasets\Celeba\img_align_celeba\041989.jpg"
save_to = r'A:\pycharm_projects\kaggle_smile_no_smile\VAE'
image = Image.open(img_path).convert("RGB")
file_name = os.path.split(img_path)[-1].split('.')[0]
input_tensor = val_transforms(image).unsqueeze(0).to(DEVICE)
output = model(input_tensor)
for i in range(2):
    predicted_img=output[i].to('cpu').detach().numpy()
    predicted_img=predicted_img.squeeze()
    predicted_img=predicted_img.transpose((1,2,0))
    predicted_img = np.clip(predicted_img, a_min=0, a_max=1)
    matplotlib.image.imsave(os.path.join(save_to,rf'{file_name}_{i}.jpg'),predicted_img)