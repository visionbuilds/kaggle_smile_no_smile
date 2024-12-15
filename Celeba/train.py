# import the necessary packages
import os
import torch
from tqdm import tqdm
import torch.optim as optim
import config, network, data_utils, model_utils
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

# create output directory
output_dir = "output"
os.makedirs("output", exist_ok=True)
# create the training_progress directory inside the output directory
training_progress_dir = os.path.join(output_dir, "training_progress")
os.makedirs(training_progress_dir, exist_ok=True)
# create the model_weights directory inside the output directory
# for storing autoencoder weights
model_weights_dir = os.path.join(output_dir, "model_weights")
os.makedirs(model_weights_dir, exist_ok=True)
# define model_weights path including best weights
MODEL_BEST_WEIGHTS_PATH = os.path.join(model_weights_dir, "best_vae_smile1.pt")
MODEL_WEIGHTS_PATH = os.path.join(model_weights_dir, "vae_smile1.pt")

# Define the transformations
train_transforms = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        # transforms.CenterCrop(148),
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor(),
    ]
)
val_transforms = transforms.Compose(
    [
        # transforms.RandomHorizontalFlip(),
        # transforms.CenterCrop(148),
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor(),
    ]
)

# Instantiate the dataset
celeba_dataset = data_utils.CelebADataset(config.DATASET_PATH, transform=train_transforms)
# Define the size of the validation set
val_size = int(len(celeba_dataset) * 0.1)  # 10% for validation
train_size = len(celeba_dataset) - val_size
train_dataset, val_dataset = random_split(celeba_dataset, [train_size, val_size])
# Define the data loaders
train_dataloader = DataLoader(
    train_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
)

model = network.CelebVAE(config.CHANNELS, config.EMBEDDING_DIM)
model = model.to(config.DEVICE)
# instantiate optimizer, and scheduler
optimizer = optim.Adam(model.parameters(), lr=config.LR)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

# initialize the best validation loss as infinity
best_val_loss = float("inf")
print(f"device {config.DEVICE}")
print("Training Started!!")
# start training by looping over the number of epochs
for epoch in range(config.EPOCHS):
    running_loss = 0.0
    for i, x in tqdm(enumerate(train_dataloader)):
        x = x.to(config.DEVICE)
        optimizer.zero_grad()
        predictions = model(x)
        total_loss = model_utils.loss_function(predictions, config.KLD_WEIGHT)
        # Backward pass
        total_loss["loss"].backward()
        # Optimizer variable updates
        optimizer.step()
        running_loss += total_loss["loss"].item()
        # compute average loss for the epoch
        train_loss = running_loss / len(train_dataloader)
        # if i > 10:
        #     break
    # compute validation loss for the epoch
    val_loss = model_utils.validate(model, val_dataloader, config.DEVICE, config.KLD_WEIGHT)
    # save best vae model weights based on validation loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(
            {"vae-celeba": model.state_dict()},
            MODEL_BEST_WEIGHTS_PATH,
        )
        torch.save(model,'best_model.pt')
    torch.save(
        {"vae-celeba": model.state_dict()},
        MODEL_WEIGHTS_PATH,
    )
    torch.save(model, 'last_model.pt')
    print(
        f"Epoch {epoch + 1}/{config.EPOCHS}, Batch {i + 1}/{len(train_dataloader)}, "
        f"Total Loss: {total_loss['loss'].detach().item():.4f}, "
        f"Reconstruction Loss: {total_loss['Reconstruction_Loss']:.4f}, "
        f"KL Divergence Loss: {total_loss['KLD']:.4f}",
        f"Val Loss: {val_loss:.4f}",
    )
    scheduler.step()