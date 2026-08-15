"""
Lightweight NAFNet-style restoration network.

Reference: Chen, L., Chu, X., Zhang, X., Sun, J. "Simple Baselines for Image
Restoration." ECCV 2022 (arXiv:2204.04676). We follow their core finding --
a plain single-stage U-Net with a "NAFBlock" (LayerNorm -> depthwise conv ->
SimpleGate -> Simplified Channel Attention, no nonlinear activations) beats
much heavier transformer baselines (e.g. Restormer, arXiv:2111.09881) at a
fraction of the compute. That efficiency-per-FLOP property is exactly what
KLA's evaluation criteria reward (quality AND H100 inference time).

We additionally need 2x super-resolution (the degraded image is spatially
half the resolution of the ground truth), so a PixelShuffle upsampling head
is appended after the NAFNet body -- this keeps the heavy lifting (denoising,
deblurring) in the compact low-resolution latent space, which is also the
cheapest place computationally to do it (standard practice in SR literature,
e.g. Real-ESRGAN, Wang et al. 2021).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, dw_expand=2, ffn_expand=2):
        super().__init__()
        dw_c = c * dw_expand
        self.norm1 = nn.GroupNorm(1, c)  # channel-wise LayerNorm equivalent
        self.conv1 = nn.Conv2d(c, dw_c, 1)
        self.conv2 = nn.Conv2d(dw_c, dw_c, 3, padding=1, groups=dw_c)
        self.sg1 = SimpleGate()
        # simplified channel attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_c // 2, dw_c // 2, 1),
        )
        self.conv3 = nn.Conv2d(dw_c // 2, c, 1)

        ffn_c = c * ffn_expand
        self.norm2 = nn.GroupNorm(1, c)
        self.conv4 = nn.Conv2d(c, ffn_c, 1)
        self.sg2 = SimpleGate()
        self.conv5 = nn.Conv2d(ffn_c // 2, c, 1)

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)))
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)))

    def forward(self, x):
        y = self.norm1(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.sg1(y)
        y = y * self.sca(y)
        y = self.conv3(y)
        x = x + y * self.beta

        y = self.norm2(x)
        y = self.conv4(y)
        y = self.sg2(y)
        y = self.conv5(y)
        x = x + y * self.gamma
        return x


class NAFNetLiteSR(nn.Module):
    """Single-stage U-Net with NAFBlocks + a 2x PixelShuffle upsample head.

    Input:  degraded image, 1xHxW (grayscale), possibly out-of-[0,1] range
            due to speckle noise.
    Output: restored image, 1x(2H)x(2W), clipped to [0,1] via sigmoid-free
            tanh-less linear head (clamped at inference time instead, so the
            network isn't forced to saturate on edge pixels).
    """
    def __init__(self, in_ch=1, out_ch=1, width=32, enc_blocks=(1, 1, 2), dec_blocks=(1, 1, 1),
                 middle_blocks=2):
        super().__init__()
        assert len(enc_blocks) == len(dec_blocks), "U-Net must be symmetric: len(dec_blocks) == len(enc_blocks)"
        self.intro = nn.Conv2d(in_ch, width, 3, padding=1)

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        c = width
        for n in enc_blocks:
            self.encoders.append(nn.Sequential(*[NAFBlock(c) for _ in range(n)]))
            self.downs.append(nn.Conv2d(c, c * 2, 2, 2))
            c *= 2

        self.middle = nn.Sequential(*[NAFBlock(c) for _ in range(middle_blocks)])

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for n in dec_blocks:
            self.ups.append(nn.Sequential(
                nn.Conv2d(c, c * 2, 1, bias=False), nn.PixelShuffle(2)))
            c //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(c) for _ in range(n)]))

        self.tail_conv = nn.Conv2d(c, width, 3, padding=1)

        # final 2x super-resolution upsample head (goes from degraded-res to
        # ground-truth-res)
        self.sr_head = nn.Sequential(
            nn.Conv2d(width, width * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.GELU(),
            nn.Conv2d(width, out_ch, 3, padding=1),
        )

        self.padder_size = 2 ** len(enc_blocks)

    def _pad(self, x):
        _, _, h, w = x.shape
        ph = (self.padder_size - h % self.padder_size) % self.padder_size
        pw = (self.padder_size - w % self.padder_size) % self.padder_size
        return F.pad(x, (0, pw, 0, ph)), h, w

    def forward(self, x):
        x, h, w = self._pad(x)
        x0 = self.intro(x)

        skips = []
        feat = x0
        for enc, down in zip(self.encoders, self.downs):
            feat = enc(feat)
            skips.append(feat)
            feat = down(feat)

        feat = self.middle(feat)

        for up, dec, skip in zip(self.ups, self.decoders, reversed(skips[-len(self.ups):])):
            feat = up(feat)
            feat = feat + skip
            feat = dec(feat)

        feat = self.tail_conv(feat) + x0
        out = self.sr_head(feat)
        out = out[:, :, :h * 2, :w * 2]
        return out


if __name__ == "__main__":
    m = NAFNetLiteSR(width=24, enc_blocks=(1, 1), dec_blocks=(1, 1), middle_blocks=1)
    n_params = sum(p.numel() for p in m.parameters())
    print(f"Params: {n_params/1e3:.1f}K")
    x = torch.randn(1, 1, 64, 64)
    y = m(x)
    print("in", x.shape, "out", y.shape)
