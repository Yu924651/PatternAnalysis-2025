import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# frist imporvement use activation function LeakyReLU rater than ReLU
# second imporvement add in a batch normalization
class DoubleConv(nn.Module):
    """
    Conv → ReLU → Conv → LeakyReLU
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(out_ch)
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(out_ch)
        self.relu2 = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu2(x)
        return x

class Encoder(nn.Module):
    """
    DoubleConv + downsample (stride-2 conv step)
    """
    def __init__(self, c_in, c_out):
        super().__init__()
        self.block = DoubleConv(c_in, c_out)
        self.down = nn.Conv2d(c_out, c_out, kernel_size=3, stride=2,
                              padding=1, bias=False)

    def forward(self, x):
        s = self.block(x)   # skip connection
        d = self.down(s)    # downsampled output
        return d, s

class Decoder(nn.Module):
    """
    Up → concat skip → DoubleConv
    """
    def __init__(self, c_in, c_skip, c_out):
        super().__init__()
        self.up = nn.ConvTranspose2d(c_in, c_out, kernel_size=2, stride=2)
        self.block = DoubleConv(c_out + c_skip, c_out)

    def forward(self, x, skip):
        x = self.up(x)
        # Fix size mismatch if needed
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear",
                              align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)
    
class Improved2DUNet(nn.Module):
    """
    Same U-Net structure as before, just
    DoubleConv blocks instead of ResBlocks
    """
    def __init__(self, in_channels=1, n_classes=6, base=32):
        super().__init__()
        C1, C2, C3, C4, C5 = base, base*2, base*4, base*8, base*16

        # Encoder
        self.stem = DoubleConv(in_channels, C1)
        self.enc1 = Encoder(C1, C2)
        self.enc2 = Encoder(C2, C3)
        self.enc3 = Encoder(C3, C4)

        # Bottleneck
        self.bott = DoubleConv(C4, C5)

        # Decoder
        self.dec3 = Decoder(C5, C4, C4)
        self.dec2 = Decoder(C4, C3, C3)
        self.dec1 = Decoder(C3, C2, C2)
        self.dec0 = Decoder(C2, C1, C1)

        # Final output conv (logits)
        self.head = nn.Conv2d(C1, n_classes, kernel_size=1)

    def forward(self, x):
        x0 = self.stem(x)
        x1, s1 = self.enc1(x0)
        x2, s2 = self.enc2(x1)
        x3, s3 = self.enc3(x2)

        xb = self.bott(x3)

        y3 = self.dec3(xb, s3)
        y2 = self.dec2(y3, s2)
        y1 = self.dec1(y2, s1)
        y0 = self.dec0(y1, x0)

        return self.head(y0)   # raw logits