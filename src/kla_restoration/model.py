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


def self_ensemble(model, x):
    """x8 geometric self-ensemble (IDEA F).

    Restores the input under all 8 symmetries of the square, maps each result
    back to the original orientation, and averages. Because the network is
    only approximately equivariant to those symmetries, the errors of the 8
    passes are partly independent and averaging cancels some of them --
    reliably worth a few tenths of a dB in the SR literature (Timofte et al.,
    "Seven ways to improve example-based single image super resolution",
    CVPR 2016).

    The transform must be applied to the INPUT and inverted on the OUTPUT.
    Flips are their own inverse; a rotation by k must be undone by -k, and the
    output lives at 2x scale, so the inverse acts on the upsampled grid.

    Costs 8 forward passes. On an H100 that is still milliseconds, which makes
    it a clean quality/speed dial rather than an unconditional cost.
    """
    outs = []
    for k in range(4):
        for flip in (False, True):
            y = torch.rot90(x, k, dims=(2, 3))
            if flip:
                y = torch.flip(y, dims=(3,))
            with torch.no_grad():
                out = model(y)
            # Undo, in reverse order.
            if flip:
                out = torch.flip(out, dims=(3,))
            out = torch.rot90(out, -k, dims=(2, 3))
            outs.append(out)
    return torch.stack(outs).mean(0)


class LocalGroupNorm(nn.Module):
    """GroupNorm(1, C) whose statistics are computed over a LOCAL window.

    Test-time Local Converter (TLC), Chu et al., "Improving Image Restoration
    by Revisiting Global Information Aggregation" (arXiv:2112.04491).

    THE PROBLEM. nn.GroupNorm(1, C) is LayerNorm-equivalent: it normalizes over
    channels AND every spatial position, so its statistics depend on the size
    of the input. This network trains on 64x64 patches but infers on full
    images, so at inference each block sees statistics gathered over an area
    tens of times larger than it ever saw in training -- a silent train/test
    mismatch (defect W-07).

    THE FIX. At inference, compute mean/variance over a sliding window whose
    area matches the training patch, rather than over the whole image. Training
    is untouched; this module is swapped in only for evaluation, so it changes
    no learned parameter.

    Implemented with box filters via avg_pool2d, which makes it O(N) in pixels
    rather than O(N * window^2).
    """

    def __init__(self, num_channels, window=64, eps=1e-5):
        super().__init__()
        self.window = window
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

    @staticmethod
    def _box_mean(x, k):
        """Windowed mean over (C,H,W), reflect-padded so edges stay valid."""
        pad = k // 2
        # Average over channels first (GroupNorm(1,C) pools channels too), then
        # spatially -- equivalent, and far cheaper.
        x = x.mean(dim=1, keepdim=True)
        x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        return F.avg_pool2d(x, k, stride=1)

    def forward(self, x):
        _, _, h, w = x.shape
        k = min(self.window, h, w)
        if k % 2 == 0:
            k -= 1                      # odd window keeps the output aligned
        if k < 3:                       # too small to localize; fall back
            var, mean = torch.var_mean(x, dim=(1, 2, 3), keepdim=True, unbiased=False)
        else:
            mean = self._box_mean(x, k)
            mean_sq = self._box_mean(x * x, k)
            var = (mean_sq - mean * mean).clamp_min(0)
        return (x - mean) / torch.sqrt(var + self.eps) * self.weight + self.bias

    @classmethod
    def from_group_norm(cls, gn, window=64):
        """Build a LocalGroupNorm carrying an existing GroupNorm's parameters."""
        m = cls(gn.num_channels, window=window, eps=gn.eps)
        # Inherit device/dtype from the module being replaced, so apply_tlc()
        # is safe to call before OR after model.to(device).
        m = m.to(device=gn.weight.device, dtype=gn.weight.dtype)
        with torch.no_grad():
            m.weight.copy_(gn.weight.view(1, -1, 1, 1))
            m.bias.copy_(gn.bias.view(1, -1, 1, 1))
        return m


def apply_tlc(model, window=64):
    """Swap every GroupNorm(1,C) for its local-window equivalent, in place.

    Call on an eval-mode model only. Returns the model for chaining.
    """
    for module in model.modules():
        for name, child in list(module.named_children()):
            if isinstance(child, nn.GroupNorm) and child.num_groups == 1:
                setattr(module, name, LocalGroupNorm.from_group_norm(child, window))
    return model


class DegradationEstimator(nn.Module):
    """Predicts degradation severity from the input, for FiLM conditioning (IDEA A).

    KLA's test set is explicitly OUT OF DISTRIBUTION, so the network will meet
    noise and blur levels it never trained on. A fixed-weight network has to
    average over everything it saw; one that can SEE how degraded its input is
    can adapt its response.

    This is a tiny CNN that regresses a low-dimensional degradation code, which
    the NAFBlocks then consume as per-channel scale/shift (FiLM: Perez et al.,
    "FiLM: Visual Reasoning with a General Conditioning Layer", AAAI 2018).
    Global average pooling makes the code resolution-independent, so it behaves
    the same on a 64x64 training patch and a 512x512 test image.

    Two cheap, physically-motivated summary statistics are fed in alongside the
    image, because they are exactly what distinguishes the degradations:
      * high-frequency energy  -> falls as BLUR increases
      * local variance          -> rises as NOISE increases
    """

    def __init__(self, in_ch=1, code_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch + 2, 16, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(32, code_dim), nn.GELU())

    @staticmethod
    def _stats(x):
        """Per-pixel blur and noise cues, as extra input channels."""
        blurred = F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), 3, stride=1)
        highfreq = (x - blurred).abs()                       # falls with blur
        local_var = (F.avg_pool2d(F.pad(x * x, (1, 1, 1, 1), mode="reflect"), 3, stride=1)
                     - blurred * blurred).clamp_min(0)       # rises with noise
        return torch.cat([x, highfreq, local_var], dim=1)

    def forward(self, x):
        return self.head(self.net(self._stats(x)))


class LogSpeckleBranch(nn.Module):
    """Homomorphic (log-domain) front end for multiplicative speckle (IDEA B).

    Speckle is MULTIPLICATIVE: y = x + x*n = x*(1+n). Convolutional networks,
    and the L1/SSIM losses used here, are built around ADDITIVE structure, so
    the noise magnitude varying with local brightness is exactly the case they
    handle worst.

    The homomorphic transform fixes this analytically:

        log(y) = log(x) + log(1 + n)

    which turns the multiplicative corruption into an additive one -- the form
    every standard denoiser is good at. This is the classical basis of SAR
    despeckling (Arsenault & April, JOSA 1976; Jain, "Fundamentals of Digital
    Image Processing", 1989) transplanted to semiconductor inspection.

    It also explains the property the problem statement flags three times --
    that degraded values may EXCEED the ground-truth range. Under a
    multiplicative model that is expected, not an artifact, and log1p maps
    those out-of-range values back into a compact, well-conditioned range
    instead of letting them dominate the first convolution.

    The branch is concatenated with the linear-domain input rather than
    replacing it, so the network keeps direct access to the raw signal and can
    learn how much to rely on each.
    """

    def __init__(self, out_ch=8, eps=1e-3):
        super().__init__()
        self.eps = eps
        self.proj = nn.Conv2d(1, out_ch, 3, padding=1)

    def forward(self, x):
        # clamp_min keeps log1p finite for the negative values additive read
        # noise can produce; eps avoids log(0).
        return self.proj(torch.log1p(x.clamp_min(0.0) + self.eps))


class NAFBlock(nn.Module):
    def __init__(self, c, dw_expand=2, ffn_expand=2, code_dim=0):
        super().__init__()
        # FiLM conditioning (IDEA A): a degradation code modulates this block
        # via per-channel scale and shift. Zero-initialized so an untrained
        # conditioning path starts as the identity and cannot destabilize
        # training.
        self.film = None
        if code_dim:
            self.film = nn.Linear(code_dim, c * 2)
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)
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

    def forward(self, x, code=None):
        if self.film is not None and code is not None:
            scale, shift = self.film(code).chunk(2, dim=1)
            x = x * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
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


class NAFStage(nn.ModuleList):
    """A run of NAFBlocks that forwards an optional conditioning code.

    nn.Sequential cannot pass a second argument, so stages are held in a
    ModuleList with an explicit forward.
    """

    def forward(self, x, code=None):
        for blk in self:
            x = blk(x, code)
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
                 middle_blocks=2, code_dim=0, log_branch=0):
        super().__init__()
        assert len(enc_blocks) == len(dec_blocks), "U-Net must be symmetric: len(dec_blocks) == len(enc_blocks)"

        # Every argument needed to rebuild this network, so a checkpoint can
        # carry its own architecture instead of relying on call sites to
        # hardcode matching values (see .config / .from_checkpoint below).
        self.config = {
            "in_ch": in_ch,
            "out_ch": out_ch,
            "width": width,
            "enc_blocks": tuple(enc_blocks),
            "dec_blocks": tuple(dec_blocks),
            "middle_blocks": middle_blocks,
            "code_dim": code_dim,
            "log_branch": log_branch,
        }

        # IDEA A: degradation estimator producing a FiLM code.
        self.estimator = DegradationEstimator(in_ch, code_dim) if code_dim else None
        # IDEA B: homomorphic log-domain branch, concatenated with the input.
        self.log_branch = LogSpeckleBranch(log_branch) if log_branch else None

        self.intro = nn.Conv2d(in_ch + log_branch, width, 3, padding=1)

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        c = width
        for n in enc_blocks:
            self.encoders.append(NAFStage(NAFBlock(c, code_dim=code_dim) for _ in range(n)))
            self.downs.append(nn.Conv2d(c, c * 2, 2, 2))
            c *= 2

        self.middle = NAFStage(NAFBlock(c, code_dim=code_dim) for _ in range(middle_blocks))

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for n in dec_blocks:
            self.ups.append(nn.Sequential(
                nn.Conv2d(c, c * 2, 1, bias=False), nn.PixelShuffle(2)))
            c //= 2
            self.decoders.append(NAFStage(NAFBlock(c, code_dim=code_dim) for _ in range(n)))

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

    # ------------------------------------------------------ checkpoint I/O --
    @classmethod
    def from_checkpoint(cls, ckpt, strict=True):
        """Rebuild a model from a checkpoint dict, using its stored config.

        This is the single place that knows how to reconstruct the network, so
        callers never hardcode architecture arguments that could drift out of
        sync with the saved weights.

        Accepts checkpoints written by scripts/train.py:
            {"model": state_dict, "config": {...}}
        and also legacy checkpoints that stored only {"model", "width"}, whose
        architecture is inferred from the state dict itself.
        """
        state = ckpt["model"]
        config = ckpt.get("config")

        if config is None:
            # Legacy checkpoint: recover the architecture from the weights.
            config = cls.infer_config(state, width=ckpt.get("width"))

        model = cls(**config)
        model.load_state_dict(state, strict=strict)
        return model

    @staticmethod
    def infer_config(state, width=None):
        """Derive constructor arguments from a state dict.

        Used for legacy checkpoints that predate the stored `config` key.
        Counts encoder/decoder stages and blocks from parameter names rather
        than assuming a fixed topology.
        """
        if width is None:
            width = state["intro.weight"].shape[0]

        def _stage_count(prefix):
            return len({k.split(".")[1] for k in state if k.startswith(f"{prefix}.")})

        def _blocks_per_stage(prefix, n_stages):
            counts = []
            for i in range(n_stages):
                counts.append(len({
                    k.split(".")[2] for k in state if k.startswith(f"{prefix}.{i}.")
                }))
            return tuple(counts)

        n_enc = _stage_count("encoders")
        n_dec = _stage_count("decoders")
        n_mid = len({k.split(".")[1] for k in state if k.startswith("middle.")})

        return {
            "in_ch": 1,
            "out_ch": state["sr_head.3.weight"].shape[0],
            "width": width,
            "enc_blocks": _blocks_per_stage("encoders", n_enc),
            "dec_blocks": _blocks_per_stage("decoders", n_dec),
            "middle_blocks": n_mid,
            "code_dim": state["estimator.head.1.weight"].shape[0] if "estimator.head.1.weight" in state else 0,
            "log_branch": state["log_branch.proj.weight"].shape[0] if "log_branch.proj.weight" in state else 0,
        }

    def _pad(self, x):
        _, _, h, w = x.shape
        ph = (self.padder_size - h % self.padder_size) % self.padder_size
        pw = (self.padder_size - w % self.padder_size) % self.padder_size
        return F.pad(x, (0, pw, 0, ph)), h, w

    def forward(self, x):
        x, h, w = self._pad(x)

        # Degradation code is estimated from the RAW input, before any
        # feature extraction, so it describes the input's corruption rather
        # than the network's internal state.
        code = self.estimator(x) if self.estimator is not None else None

        if self.log_branch is not None:
            x = torch.cat([x, self.log_branch(x)], dim=1)

        x0 = self.intro(x)

        skips = []
        feat = x0
        for enc, down in zip(self.encoders, self.downs):
            feat = enc(feat, code)
            skips.append(feat)
            feat = down(feat)

        feat = self.middle(feat, code)

        for up, dec, skip in zip(self.ups, self.decoders, reversed(skips[-len(self.ups):])):
            feat = up(feat)
            feat = feat + skip
            feat = dec(feat, code)

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
