import torch
import torch.nn as nn

class DoubleConv(nn.Module):
   """(convolution => [BN] => ReLU) * 2"""

   def __init__(self, in_channels, out_channels, mid_channels=None):
       super().__init__()
       if not mid_channels:
           mid_channels = out_channels
       self.double_conv = nn.Sequential(
           nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
           nn.BatchNorm2d(mid_channels),
           nn.LeakyReLU(0.2, inplace=True),
           nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
           nn.BatchNorm2d(out_channels),
           nn.LeakyReLU(0.2, inplace=True)
       )

   def forward(self, x):
       return self.double_conv(x)

class Down(nn.Module):
   """Downscaling with maxpool then double conv"""

   def __init__(self, in_channels, out_channels):
       super().__init__()
       self.maxpool_conv = nn.Sequential(
           nn.MaxPool2d(2),
           DoubleConv(in_channels, out_channels)
       )

   def forward(self, x):
       return self.maxpool_conv(x)

class OutConv(nn.Module):
   def __init__(self, in_channels, out_channels):
       super(OutConv, self).__init__()
       self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

   def forward(self, x):
       return self.conv(x)

class MyDiscriminator(nn.Module):
   def __init__(self, n_channels=2, n_classes=1):
       super().__init__()
       self.n_channels = n_channels
       self.n_classes = n_classes

       self.inc = DoubleConv(n_channels, 4)
       self.down0 = Down(4, 8)
       self.down1 = Down(8, 16)
       self.down2 = Down(16, 32)
       self.down3 = Down(32, 64)
       self.down4 = Down(64, 128)
       self.down5 = Down(128, 256)
       self.outc = OutConv(256, n_classes)


   def forward(self, x):
       x = self.inc(x)
       x = self.down0(x)
       x = self.down1(x)
       x = self.down2(x)
       x = self.down3(x)
       x = self.down4(x)
       x = self.down5(x)
       logits = self.outc(x)
       return logits


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = self.conv_bn_relu(4, 16, kernel_size=5, reps=1) # fake/true color + gray
        self.conv2 = self.conv_bn_relu(16, 32, pool_kernel=4)
        self.conv3 = self.conv_bn_relu(32, 64, pool_kernel=2)
        self.conv4 = self.conv_bn_relu(64, 128, pool_kernel=2)
        self.conv5 = self.conv_bn_relu(128, 256, pool_kernel=2)
        self.out_patch = nn.Conv2d(256, 1, kernel_size=1) #1x8x8

    def conv_bn_relu(self, in_ch, out_ch, kernel_size=3, pool_kernel=None, reps=2):
        layers = []
        for i in range(reps):
            if i == 0 and pool_kernel is not None:
                layers.append(nn.AvgPool2d(pool_kernel))
            layers.append(nn.Conv2d(in_ch if i == 0 else out_ch,
                                    out_ch, kernel_size, padding=(kernel_size - 1) // 2))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        out = self.out_patch(x)
        return out
