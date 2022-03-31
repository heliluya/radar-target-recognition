""" Full assembly of the parts to form the complete network """

from .unet_parts import *
from .vit import Transformer


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv3d(n_channels, 64)
        self.squeeze_speed1 = Transformer(n_patches=32, hidden_size=64, dropout_rate=0.1)
        self.down1 = Down(64, 128)
        self.squeeze_speed2 = Transformer(n_patches=16, hidden_size=128, dropout_rate=0.1)
        self.down2 = Down(128, 256)
        self.squeeze_speed3 = Transformer(n_patches=8, hidden_size=256, dropout_rate=0)
        self.down3 = Down(256, 512)
        self.squeeze_speed4 = Transformer(n_patches=4, hidden_size=512, dropout_rate=0.0)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.squeeze_speed = ConnectEncoderDecoder(1024 * 2, 1024, 2)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # x的速度轴size必须为64
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.squeeze_speed(x5)
        x4 = self.squeeze_speed4(x4)
        x = self.up1(x5, x4)
        x3 = self.squeeze_speed3(x3)
        x = self.up2(x, x3)
        x2 = self.squeeze_speed2(x2)
        x = self.up3(x, x2)
        x1 = self.squeeze_speed1(x1)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
