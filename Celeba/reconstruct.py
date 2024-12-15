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
import config, network, data_utils, model_utils

DEVICE = ( "cuda" if torch.cuda.is_available() else "cpu")
model_path = r"A:\pycharm_projects\kaggle_smile_no_smile\Celeba\best_model.pt"
model=torch.load(model_path,map_location=DEVICE)
model.eval()
val_transforms = transforms.Compose(
    [
        # transforms.RandomHorizontalFlip(),
        # transforms.CenterCrop(148),
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor(),
    ]
)

# img_path=r"F:\pycharm_projects_personal\smile_not_smile\VAE\data\Celeba\kaggle\train\Albrecht_Mentz_0001.jpg"
img_path=r"A:\pycharm_projects\kaggle_smile_no_smile\datasets\from_kaggle\valid\non_smile\Aaron_Eckhart_0001.jpg"
save_to = r'A:\pycharm_projects\kaggle_smile_no_smile\Celeba\output\reconstruction'
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
