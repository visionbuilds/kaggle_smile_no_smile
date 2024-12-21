import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm
import os
from torchsummary import summary
# import torch
print(torch.cuda.is_available())  # Should return True
print(torch.cuda.get_device_name(0))
print(torch.version.cuda)
# Sampling function
def sample_latent(mean, log_variance):
    std = torch.exp(0.5 * log_variance)
    epsilon = torch.randn_like(std)
    return mean + epsilon * std

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
        latent=sample_latent(mean, log_variance)
        return mean, log_variance,latent

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
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.fc(x)
        x = self.unflatten(x)
        x = self.decoder(x)
        return x



# Loss functions

def reconstruction_loss(y_true, y_pred, img_width, img_height):

    mse_loss = nn.MSELoss(reduction='sum')(y_pred, y_true)  # Mean MSE over all pixels
    # return mse_loss * img_width * img_height
    return mse_loss

def kl_loss(mean, log_variance):
    return -0.5 * torch.sum(1 + log_variance - mean.pow(2) - log_variance.exp())

def vae_loss(y_true, y_pred, mean, log_variance,epoch,img_width, img_height):
    recon_loss = reconstruction_loss(y_true, y_pred,img_width, img_height)
    kl = kl_loss(mean, log_variance)
    if epoch % 50 == 0:
        print(f"recon_loss {recon_loss}, kl loss {kl}")
    return recon_loss + 0.01*kl



# Parameters
img_height, img_width = 128, 128
latent_dim = 200
epochs=500
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device {device}")
transform = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    # transforms.RandomVerticalFlip(p=0.2
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])
# Initialize the model, optimizer, and dataloader
encoder = Encoder(img_height, img_width, latent_dim).to(device)
decoder = Decoder(latent_dim, img_height, img_width).to(device)
print(summary(decoder, input_size=(200,)))
optimizer_enc = optim.Adam(encoder.parameters(), lr=0.0005)
optimizer_dec = optim.Adam(decoder.parameters(), lr=0.0005)

# Replace `your_dataloader` with your dataset DataLoader
# main_path = r'./datasets/from_kaggle/train_and_valid'
main_path = r'/media/yuri/A/pycharm_projects/kaggle_smile_no_smile/datasets/from_kaggle/train_and_valid'
save_dir=r'/media/yuri/A/pycharm_projects/kaggle_smile_no_smile/Celeba/output/reconstruction'
dataset = datasets.ImageFolder(root=main_path, transform=transform)

val_size = int(len(dataset) * 0.1)  # 10% for validation
train_size = len(dataset) - val_size
train_dataset, valid_dataset = random_split(dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=2)
valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=2)


encoder.train()
decoder.train()
losses = []
valid_losses= []
best_encoder=None
best_decoder=None
best_loss=10000000000
best_epoch=0
t_losses = []
b_losses = []
for epoch in tqdm(range(epochs)):
    batch_losses = []
    for images, _ in train_dataloader:
        train_images=images.to(device)
        mean, log_variance,latent = encoder(train_images)
        if torch.isnan(mean).any() or torch.isinf(mean).any():
            print("NaN or Inf detected in mean")
            break

        if torch.isnan(log_variance).any() or torch.isinf(log_variance).any():
            print("NaN or Inf detected in log_variance")
            break

        reconstructed = decoder(latent)
        loss = vae_loss(train_images, reconstructed, mean, log_variance,epoch,img_width, img_height)
        b_losses.append(loss)
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
    train_epoch_loss = sum(batch_losses) / len(batch_losses)
    losses.append(train_epoch_loss)

    batch_losses = []
    with torch.no_grad():
        for images, _ in valid_dataloader:
            valid_images=images.to(device)
            mean, log_variance,latent = encoder(valid_images)
            reconstructed = decoder(latent)
            loss = vae_loss(valid_images, reconstructed, mean, log_variance, epoch, img_width, img_height)
            batch_losses.append(loss.item())
    valid_epoch_loss = sum(batch_losses) / len(batch_losses)
    valid_losses.append(valid_epoch_loss)

    if epoch % 10 ==0:
        with torch.no_grad():
            encoder.eval()
            decoder.eval()
            # save_dir=r'./Celeba/output/reconstruction'
            big_combined_img=None
            for i in range(len(valid_images)):
                mean, log_variance, latent = encoder(valid_images)
                reconstructed = decoder(latent)
                predicted_img = reconstructed[i].to('cpu').detach().numpy()
                predicted_img = predicted_img.squeeze()
                predicted_img = predicted_img.transpose((1, 2, 0))
                predicted_img = np.clip(predicted_img, a_min=0, a_max=1)

                original_image = valid_images[i]
                original_image = original_image.to('cpu').detach().numpy()
                original_image = original_image.transpose((1, 2, 0))
                original_image = np.clip(original_image, a_min=0, a_max=1)
                combined_img = np.concatenate([original_image, predicted_img], axis=0)
                if big_combined_img is None:
                    big_combined_img = combined_img
                else:
                    big_combined_img = np.concatenate([big_combined_img,combined_img],axis=1)
            matplotlib.image.imsave(os.path.join(save_dir, f"{epoch}.jpg"), big_combined_img)

    # if epoch_loss < best_loss and epoch > 150 and (epoch - best_epoch) > 50:
    #     best_epoch = epoch
    #     best_loss = epoch_loss
    #     best_encoder = encoder
    #     best_decoder = decoder
    #     print(f"epoch {epoch} save best models")
    # # if epoch % 10 == 0:
    # #     print(f"Epoch {epoch}, Loss: {epoch_loss}")

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

def draw_two_lines(list1, list2, img_path):
    """
    Draws two lines on the same graph from two lists of float values and saves the graph as a JPG file.

    :param list1: List of float values for the first line.
    :param list2: List of float values for the second line.
    :param img_path: Path to save the graph as a JPG file.
    """
    if len(list1) != len(list2):
        raise ValueError("Both lists must have the same length.")

    x_values = range(len(list1))

    plt.plot(x_values, list1, label='Line 1', linestyle='-', marker='o')
    plt.plot(x_values, list2, label='Line 2', linestyle='--', marker='s')

    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.title('Two Lines from Lists')
    plt.legend()
    plt.grid(True)

    plt.savefig(img_path, format='jpg')
    plt.show()

# find average latent for smiling and non smiling images
dim = 200
avg_image_with_smile = np.zeros(dim, dtype='float32')
avg_image_without_smile = np.zeros(dim, dtype='float32')
n_images_with_smile = 0
n_images_without_smile = 0
dataloader = DataLoader(train_dataset, batch_size=1)
for images, targets in dataloader:
    train_images = images.to(device)
    mean, log_variance, latent = encoder(train_images)
    if targets.numpy()[0]==1:
        avg_image_with_smile +=latent.detach().cpu().numpy()[0]
        n_images_with_smile +=1
    else:
        avg_image_without_smile += latent.detach().cpu().numpy()[0]
        n_images_without_smile += 1
avg_image_with_smile /= n_images_with_smile
avg_image_without_smile /= n_images_without_smile



# add smile to images
diff = (avg_image_with_smile - avg_image_without_smile)
diff = torch.tensor(diff[None,:]).to(device)
encoder.eval()
decoder.eval()
test_dataloader = DataLoader(valid_dataset, batch_size=1)
diff_coefficients = [0, 0.5,1., 1.5, 2.5]
save_dir=r'/media/yuri/A/pycharm_projects/kaggle_smile_no_smile/Celeba/output/added_smile'
for k,(images, targets) in enumerate(test_dataloader):
    if targets.numpy()[0] == 1:
        continue
    fig, axes = plt.subplots(nrows=2, ncols=len(diff_coefficients), sharex=True, sharey=True, figsize=(8, 3.5))
    images = images.to(device)
    mean, log_variance, latent = encoder(images)
    # latent = latent.detach().cpu().numpy()[0]
    for i, alpha in enumerate(diff_coefficients):
        more_smile = latent + alpha * diff
        less_smile = latent - alpha * diff
        more = decoder(more_smile)
        less = decoder(less_smile)
        more = more.detach().cpu().numpy()[0].transpose([1, 2, 0])
        less = less.detach().cpu().numpy()[0].transpose([1, 2, 0])
        axes[0, i].set_title(alpha)
        axes[0, i].imshow(more)
        axes[1, i].imshow(less)
    plt.savefig(os.path.join(save_dir,f"{k}.jpg"))

