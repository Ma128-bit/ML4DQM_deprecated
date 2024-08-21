import numpy as np
import torch.nn as nn
import torch.nn.functional as F


from ResNet import ResBlock
from ResNet import ResBlockTranspose
import torch


class ResNetVAE(nn.Module):
    """
    Define the full ResNet autoencoder model
    """

    def __init__(self, in_channels, nblocks, fmaps, img_size, device, latent_size):
        super(ResNetVAE, self).__init__()

        self.fmaps = fmaps
        self.nblocks = nblocks
        self.in_channels = in_channels
        self.img_size = img_size
        self.device = device

        self.debug = False
        self.latent_size = latent_size
        self._initialize_sizes(img_size, fmaps)
        self._initialize_encoding_layers(in_channels, fmaps)
        self._initialize_decoding_layers(in_channels, fmaps)

    def _initialize_encoding_layers(self, in_channels, fmaps):
        # Initialize encoding layers
        self.encoding_layers = nn.ModuleList()

        self.econv0 = nn.Sequential(
            nn.Conv2d(in_channels, fmaps[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        for i in range(0, len(fmaps)):
            self.encoding_layers.append(
                self.block_layers(self.nblocks, [fmaps[i], fmaps[i]], "enc")
            )
            if i != len(fmaps) - 1:
                self.encoding_layers.append(
                    self.block_layers(1, [fmaps[i], fmaps[i + 1]], "enc")
                )

        self.mean_layer = nn.Linear(
            self.fmaps[-1] * self.sizes[-1][0] * self.sizes[-1][1], self.latent_size
        )
        self.logvar_layer = nn.Linear(
            self.fmaps[-1] * self.sizes[-1][0] * self.sizes[-1][1], self.latent_size
        )

    def _initialize_sizes(self, img_size, fmaps):
        self.sizes = [list(img_size)]
        self.sizes.append([int(np.floor(el * 0.5)) for el in self.sizes[0]])
        for i in range(len(fmaps) - 1):
            self.sizes.append([int(np.ceil(el * 0.5)) for el in self.sizes[-1]])
        if self.debug:
            print(f"sizes: {self.sizes}")

    def _initialize_decoding_layers(self, in_channels, fmaps):
        # Initialize decoding layers
        self.decoding_layers = nn.ModuleList()
        self.fc = nn.Linear(
            self.latent_size, self.fmaps[-1] * self.sizes[-1][0] * self.sizes[-1][1]
        )
        for i in range(len(fmaps) - 1, -1, -1):
            self.decoding_layers.append(
                self.block_layers(self.nblocks, [fmaps[i], fmaps[i]], "dec")
            )
            if i != 0:
                self.decoding_layers.append(
                    self.block_layers(
                        1, [fmaps[i], fmaps[i - 1]], "dec", out_shape=self.sizes[i]
                    )
                )

        self.dconv0 = nn.ConvTranspose2d(
            fmaps[0], in_channels, kernel_size=3, stride=1, padding=(1, 1)
        )
        self.dconv0_relu = nn.ReLU(inplace=True)

    def _reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(self.device)
        z = mean + var * epsilon
        return z

    def block_layers(self, nblocks, fmaps, state, out_shape=None):
        """
        Convenience function: append several resnet blocks in sequence
        """
        layers = []
        for _ in range(nblocks):
            if state == "enc":
                layers.append(ResBlock(fmaps[0], fmaps[1]))
            else:
                layers.append(ResBlockTranspose(fmaps[0], fmaps[1], out_shape))
        return nn.Sequential(*layers)

    def forward(self, x):

        mean, logvar = self.encode(x)
        z = self._reparameterization(mean, torch.exp(0.5 * logvar))
        x_hat = self.decode(z)

        return x_hat, mean, logvar

    def encode(self, x):
        if self.debug:
            print(x.size())
        if self.debug:
            print("Encode")

        x = self.econv0(x)
        if self.debug:
            print(x.size())
        x = F.max_pool2d(x, kernel_size=2)
        if self.debug:
            print(x.size())

        for layer in self.encoding_layers:
            x = layer(x)
            if self.debug:
                print(x.size())

        # mean and var
        x = x.view(x.size()[0], -1)
        if self.debug:
            print(x.size())
        mean = self.mean_layer(x)
        logvar = self.logvar_layer(x)

        if self.debug:
            print(f"mean.size(): {mean.size()}")
            print(f"logvar.size(): {logvar.size()}")

        return mean, logvar

    def decode(self, z):
        if self.debug:
            print("Decode")

        x = self.fc(z)

        if self.debug:
            print(x.size())
        x = x.view(-1, self.fmaps[-1], self.sizes[-1][0], self.sizes[-1][1])
        if self.debug:
            print(x.size())

        for layer in self.decoding_layers:
            x = layer(x)
            if self.debug:
                print(x.size())

        # Interpolate to original size
        x = F.interpolate(x, size=list(self.sizes[0]))
        if self.debug:
            print(x.size())
        x = self.dconv0(
            x,
            output_size=(
                x.size()[0],
                self.in_channels,
                self.img_size[0],
                self.img_size[1],
            ),
        )
        x = self.dconv0_relu(x)
        if self.debug:
            print(x.size())

        return x