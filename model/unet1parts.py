# sub-parts of the U-Net model
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class double_conv1(nn.Module):
    '''(conv1 => bn => ReLU), twice'''
    def __init__(self, in_ch, out_ch):
        super(double_conv1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        return x


class inconv1(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv1, self).__init__()
        self.conv1 = double_conv1(in_ch, out_ch)

    def forward(self, x):
        x = self.conv1(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv1 = nn.Sequential(
            nn.MaxPool1d(2),
            double_conv1(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv1(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, linear=True):
        super(up, self).__init__()
        if linear:
            self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        else:
            self.up = nn.ConvTranspose1d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv1 = double_conv1(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # input is CxL
        diffL = x2.size()[2] - x1.size()[2]

        # Ensure that we pad appropriately to get the right size...
        x1 = F.pad(x1, (diffL // 2, diffL - diffL//2))

        x = torch.cat([x2, x1], dim=1)
        x = self.conv1(x)
        return x


class outconv1(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv1, self).__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv1(x)
        return x