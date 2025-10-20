import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResBlock(nn.Module):
    """Residual double-conv with BatchNorm + ReLU."""
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(inplace=True)

        # adjust skip connection if input and output channels differ
        self.skip = nn.Identity() if c_in == c_out else nn.Conv2d(c_in, c_out, kernel_size=1, bias=False)

    def forward(self, x):
        # --- main path ---
        out = self.conv1(x)          # 1st conv
        out = self.bn1(out)          # batch norm
        out = self.relu(out)         # ReLU activation

        out = self.conv2(out)        # 2nd conv
        out = self.bn2(out)          # batch norm

        # --- skip connection ---
        residual = self.skip(x)      # either Identity() or 1×1 conv

        # --- combine + activate ---
        out = out + residual         # element-wise addition
        out = self.relu(out)         # final ReLU

        return out

class DownUnit(nn.Module):
    """Residual block + stride-2 conv for downsampling."""
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.block = ResBlock(c_in, c_out)
        self.down = nn.Conv2d(c_out, c_out, kernel_size=3, stride=2, padding=1, bias=False)

    def forward(self, x):
        x = self.block(x)
        d = self.down(x)
        return d, x  # d → next stage, x → skip connection


class UpUnit(nn.Module):
    """Bilinear upsample + 1x1 reduce + concat + residual block."""
    def __init__(self, c_in: int, c_skip: int, c_out: int):
        super().__init__()
        self.reduce = nn.Conv2d(c_in, c_out, kernel_size=1, bias=False)
        self.block = ResBlock(c_out + c_skip, c_out)
    
    def forward(self, x, skip):
        # --- upsample decoder feature map ---
        x = F.interpolate(
            x,
            scale_factor=2.0,
            mode="bilinear",
            align_corners=False
        )

        # --- reduce channel depth after upsampling ---
        x = self.reduce(x)

        # --- make sure spatial size matches the skip connection ---
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(
                x,
                size=skip.shape[2:],   # resize to match height & width of skip
                mode="bilinear",
                align_corners=False
            )

        # --- concatenate encoder skip with upsampled decoder feature ---
        x = torch.cat([x, skip], dim=1)

        # --- refine combined feature with a residual block ---
        x = self.block(x)

        return x
    
class imporved2DUNet(nn.Module):
    def __init__(self, in_channels=1, n_classes=6, base=32):
        super(imporved2DUNet, self).__init__()
        # Encoder
        C1, C2, C3, C4, C5 = base, base*2, base*4, base*8, base*16

        # Encoder
        self.stem = ResBlock(in_channels, C1)
        self.enc1 = DownUnit(C1, C2)
        self.enc2 = DownUnit(C2, C3)
        self.enc3 = DownUnit(C3, C4)

        # Bottleneck
        self.bott = ResBlock(C4, C5)

        # Decoder
        self.dec3 = UpUnit(C5, C4, C4)
        self.dec2 = UpUnit(C4, C3, C3)
        self.dec1 = UpUnit(C3, C2, C2)
        self.dec0 = UpUnit(C2, C1, C1)

        # Head
        self.head = nn.Conv2d(C1, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x0 = self.stem(x)        # (B,C1,H,W)
        x1, s1 = self.enc1(x0)   # (B,C2,H/2,W/2), skip s1=(B,C2,H/2,W/2)
        x2, s2 = self.enc2(x1)   # (B,C3,H/4,W/4), skip s2
        x3, s3 = self.enc3(x2)   # (B,C4,H/8,W/8), skip s3

        # Bottleneck
        xb = self.bott(x3)       # (B,C5,H/8,W/8)

        # Decoder
        y3 = self.dec3(xb, s3)   # -> (B,C4,H/4,W/4)
        y2 = self.dec2(y3, s2)   # -> (B,C3,H/2,W/2)
        y1 = self.dec1(y2, s1)   # -> (B,C2,H,W)
        y0 = self.dec0(y1, x0)   # -> (B,C1,H,W)   (note: last skip is stem output)

        return self.head(y0)     # (B,n_classes,H,W)



if __name__ == "__main__":
    model = imporved2DUNet(in_channels=1, n_classes=6, base=32)
    x = torch.randn(8, 1, 255, 321)  # batch_size=8; odd H/W to test alignment
    with torch.no_grad():
        y = model(x)
    print("Input :", tuple(x.shape))
    print("Output:", tuple(y.shape))  # should be (8, 6, 255, 321)
    print("Params:", sum(p.numel() for p in model.parameters()))