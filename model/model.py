# full assembly of the sub-parts to form the complete net

import torch.nn.functional as F
import torch as T

from .unet1parts import *


class UNet1SnC(nn.Module):
    # A simplified version of a 1D UNet architecture with a modest number of channels
    def __init__(self, n_channels, n_classes):
        super(UNet1SnC, self).__init__()
        self.inc = inconv1(n_channels, 16)
        self.down1 = down(16, 32)
        self.down2 = down(32, 64)
        self.down3 = down(64, 128)
        self.down4 = down(128, 128)
        self.up1 = up(256, 64)
        self.up2 = up(128, 32)
        self.up3 = up(64, 16)
        self.up4 = up(32, 16)
        self.outc = outconv1(16, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return T.sigmoid(x), T.argmax(x, dim=1) # NOT Differentiable
