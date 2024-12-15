# import the necessary packages
import matplotlib
import numpy as np
from PIL import Image
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

def loss_function(VAELossParams, kld_weight):
    recons, input, mu, log_var = VAELossParams
    recons_loss = F.mse_loss(recons, input)
    kld_loss = torch.mean(
        -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
    )
    loss = recons_loss + kld_weight * kld_loss
    return {
        "loss": loss,
        "Reconstruction_Loss": recons_loss.detach(),
        "KLD": -kld_loss.detach(),
    }

def validate(model, val_dataloader, device,kld_weight):
    running_loss = 0.0
    with torch.no_grad():
        # Trace
        print('Running Validation')
        # for batch_idx, (data, _) in enumerate(val_dataloader):
        for data in val_dataloader:
            # Convert to cuda if possible
            data = data.to(device=device)

            # Forward
            predictions = model(data)
            loss = loss_function(predictions, kld_weight)
            loss = loss["loss"].item()
            running_loss +=loss
    total_loss = running_loss/len(val_dataloader)
    return total_loss