# import the necessary packages
from typing import List
import torch
import torch.nn as nn
from torch import Tensor

# define a convolutional block for the encoder part of the vae
class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(ConvBlock, self).__init__()
        # sequential block consisting of a 2d convolution,
        # batch normalization, and leaky relu activation
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,  # number of input channels
                out_channels=out_channels,  # number of output channels
                kernel_size=3,  # size of the convolutional kernel
                stride=2,  # stride of the convolution
                padding=1,  # padding added to the input
            ),
            nn.BatchNorm2d(out_channels),  # normalize the activations of the layer
            nn.LeakyReLU(),  # apply leaky relu activation
        )
    def forward(self, x):
        # pass the input through the sequential block
        return self.block(x)

# define a transposed convolutional block for the decoder part of the vae
class ConvTBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(ConvTBlock, self).__init__()
        # sequential block consisting of a transposed 2d convolution,
        # batch normalization, and leaky relu activation
        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,  # number of input channels
                out_channels,  # number of output channels
                kernel_size=3,  # size of the convolutional kernel
                stride=2,  # stride of the convolution
                padding=1,  # padding added to the input
                output_padding=1,  # additional padding added to the output
            ),
            nn.BatchNorm2d(out_channels),  # normalize the activations of the layer
            nn.LeakyReLU(),  # apply leaky relu activation
        )
    def forward(self, x):
        return self.block(x)  # pass the input through the sequential block

# define the main vae class
class CelebVAE(nn.Module):
    def __init__(
        self, in_channels: int, latent_dim: int, hidden_dims: List = None
    ) -> None:
        super(CelebVAE, self).__init__()
        self.latent_dim = latent_dim  # dimensionality of the latent space
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]  # default hidden dimensions
        # build the encoder using convolutional blocks
        self.encoder = nn.Sequential(
            *[
                # create a convblock for each pair of input and output channels
                ConvBlock(in_f, out_f)
                for in_f, out_f in zip([in_channels] + hidden_dims[:-1], hidden_dims)
            ]
        )  # fully connected layer for the mean of the latent space
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        # fully connected layer for the variance of the latent space
        self.fc_var = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        # build the decoder using transposed convolutional blocks
        # fully connected layer to expand the latent space
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)
        hidden_dims.reverse()  # reverse the hidden dimensions for the decoder
        self.decoder = nn.Sequential(
            *[
                # create a convtblock for each pair of input and output channels
                ConvTBlock(in_f, out_f)
                for in_f, out_f in zip(hidden_dims[:-1], hidden_dims[1:])
            ]
        )
        # final layer to reconstruct the original input
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dims[-1],
                hidden_dims[-1],
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            # normalize the activations of the layer
            nn.BatchNorm2d(hidden_dims[-1]),
            # apply leaky relu activation
            nn.LeakyReLU(),
            # final convolution to match the output channels
            nn.Conv2d(hidden_dims[-1], out_channels=3, kernel_size=3, padding=1),
            # apply tanh activation to scale the output
            nn.Tanh(),
        )
    # encoding function to map the input to the latent space
    def encode(self, input: Tensor) -> List[Tensor]:
        # pass the input through the encoder
        result = self.encoder(input)
        # flatten the result for the fully connected layers
        result = torch.flatten(result, start_dim=1)
        # compute the mean of the latent space
        mu = self.fc_mu(result)
        # compute the log variance of the latent space
        log_var = self.fc_var(result)
        return [mu, log_var]

    # decoding function to map the latent space to the reconstructed input
    def decode(self, z: Tensor) -> Tensor:
        # expand the latent space
        result = self.decoder_input(z)
        # reshape the result for the transposed convolutions
        result = result.view(-1, 512, 2, 2)
        # pass the result through the decoder
        result = self.decoder(result)
        # pass the result through the final layer
        result = self.final_layer(result)
        return result

    # reparameterization trick to sample from the latent space
    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        # compute the standard deviation from the log variance
        std = torch.exp(0.5 * logvar)
        # sample random noise
        eps = torch.randn_like(std)
        # compute the sample from the latent space
        return eps * std + mu

    # forward pass of the vae
    def forward(self, input: Tensor) -> List[Tensor]:
        # encode the input to the latent space
        mu, log_var = self.encode(input)
        # sample from the latent space
        z = self.reparameterize(mu, log_var)
        # decode the sample, and return the reconstruction
        # along with the original input, mean, and log variance
        return [self.decode(z), input, mu, log_var]