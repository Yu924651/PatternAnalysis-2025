# use https://colab.research.google.com/drive/1VOsZSyRhyuHLmgoqGriQk01ub4bKNmZ1?usp=sharing as base code

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# frist imporvement use activation function LeakyReLU rater than ReLU
# second imporvement add in a batch
# Bias after BN is redundant. BatchNorm learns its own affine shift/scale, so conv bias just wastes params and can add noise.
# last imporvement added in drop out 

class DoubleConv(nn.Module):
    """
    Conv → LeakyReLU → Conv → LeakyReLU
    """
    def __init__(self, in_ch, out_ch, p_drop=0.2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(out_ch)
        self.relu1 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(out_ch)
        self.relu2 = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu2(x)
        return x

# imporvement 3 Replacing MaxPooling with a learnable stride-2 convolution
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
    8 laybels week 0-7
    """
    def __init__(self, in_channels=1, n_classes=6, base=32, p_drop=0.2):
        super().__init__()
        C1, C2, C3, C4, C5 = base, base*2, base*4, base*8, base*16

        # Encoder
        self.stem = DoubleConv(in_channels, C1)
        self.enc1 = Encoder(C1, C2)
        self.enc2 = Encoder(C2, C3)
        self.enc3 = Encoder(C3, C4)

        # Bottleneck
        self.bott = DoubleConv(C4, C5)
        self.drop_bott = nn.Dropout2d(p_drop)

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
        xb = self.drop_bott(xb)

        y3 = self.dec3(xb, s3)
        y2 = self.dec2(y3, s2)
        y1 = self.dec1(y2, s1)
        y0 = self.dec0(y1, x0)

        return self.head(y0)   # raw logits



def dice_loss(pred, target, eps=1e-6):
    pred = torch.softmax(pred, dim=1)  # convert logits → probabilities
    target_onehot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
    intersection = (pred * target_onehot).sum(dim=(0, 2, 3))
    union = pred.sum(dim=(0, 2, 3)) + target_onehot.sum(dim=(0, 2, 3))
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()
    
class DiceLoss(nn.Module):
    """Wrapper so you can still do `criterion = DiceLoss()` with no args."""
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        return dice_loss(pred, target, eps=self.eps)

# -----------------------------
# Metrics: per-class Dice (logits->argmax)
# -----------------------------
@torch.no_grad()
def dice_all_class(logits, targets, num_classes=6, eps=1e-6):
    """
    Returns a list of length C with Dice for each class.
    logits: (B,C,H,W); targets: (B,H,W) int64
    """
    preds = torch.argmax(logits, dim=1)  # (B,H,W)
    dice_scores = []
    for c in range(num_classes):
        p = (preds == c).float()
        t = (targets == c).float()
        inter = (p * t).sum(dim=(1,2))
        denom = p.sum(dim=(1,2)) + t.sum(dim=(1,2))
        dice = (2 * inter + eps) / (denom + eps)
        dice_scores.append(dice.mean().item())
    return dice_scores