"""Embedded library sources, part B: models, losses, training engine.

Same convention as part A -- these strings are wrapped in r\"\"\"...\"\"\" inside a notebook
cell, so NO triple-double-quotes anywhere below. Use # comments only.
"""

# --------------------------------------------------------------------------- baselines
CRVS_MODELS_SRC = r'''
# crvs_models.py -- the four baseline 1-D segmentation networks.
# All four are standardised the way Chowdhury et al. 2024 describe (section 3.1):
# 5 levels, 64 filters in the first level, doubling thereafter. Input (B, C_in, 1024).
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def cbr(i, o, k=3, s=1):
    return nn.Sequential(nn.Conv1d(i, o, k, s, padding=k // 2, bias=False),
                         nn.BatchNorm1d(o), nn.ReLU(inplace=True))

class DoubleConv(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.b = nn.Sequential(cbr(i, o), cbr(o, o))
    def forward(self, x):
        return self.b(x)

# ------------------------------------------------------------------ UNet
class UNet1D(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=64, levels=4):
        super().__init__()
        chs = [base * (2 ** i) for i in range(levels)]
        self.inc = DoubleConv(in_ch, chs[0])
        self.downs = nn.ModuleList()
        for i in range(levels - 1):
            self.downs.append(DoubleConv(chs[i], chs[i + 1]))
        self.bott = DoubleConv(chs[-1], chs[-1] * 2)
        self.ups = nn.ModuleList()
        self.decs = nn.ModuleList()
        prev = chs[-1] * 2
        for c in reversed(chs):
            self.ups.append(nn.ConvTranspose1d(prev, c, 4, 2, 1))
            self.decs.append(DoubleConv(c * 2, c))
            prev = c
        self.head = nn.Conv1d(chs[0], out_ch, 1)
    def forward(self, x):
        skips = []
        h = self.inc(x); skips.append(h)
        for d in self.downs:
            h = d(F.max_pool1d(h, 2)); skips.append(h)
        h = self.bott(F.max_pool1d(h, 2))
        for up, dec, sk in zip(self.ups, self.decs, reversed(skips)):
            h = up(h)
            if h.shape[-1] != sk.shape[-1]:
                h = F.interpolate(h, size=sk.shape[-1], mode="linear", align_corners=False)
            h = dec(torch.cat([h, sk], 1))
        return {"wave": torch.tanh(self.head(h))}

# ------------------------------------------------------------------ LinkNet
class LinkEnc(nn.Module):
    def __init__(self, i, o, stride=2):
        super().__init__()
        self.c1 = cbr(i, o, 3, stride)
        self.c2 = nn.Sequential(nn.Conv1d(o, o, 3, 1, 1, bias=False), nn.BatchNorm1d(o))
        self.sc = nn.Sequential(nn.Conv1d(i, o, 1, stride, bias=False), nn.BatchNorm1d(o))
    def forward(self, x):
        return F.relu(self.c2(self.c1(x)) + self.sc(x))

class LinkDec(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        m = max(i // 4, 8)
        self.a = cbr(i, m, 1)
        self.b = nn.Sequential(nn.ConvTranspose1d(m, m, 4, 2, 1, bias=False),
                               nn.BatchNorm1d(m), nn.ReLU(inplace=True))
        self.c = cbr(m, o, 1)
    def forward(self, x):
        return self.c(self.b(self.a(x)))

class LinkNet1D(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=64, levels=4):
        super().__init__()
        chs = [base * (2 ** i) for i in range(levels)]
        self.stem = cbr(in_ch, chs[0], 7, 1)
        self.encs = nn.ModuleList()
        prev = chs[0]
        for c in chs:
            self.encs.append(LinkEnc(prev, c, 2)); prev = c
        self.bott = cbr(prev, prev)
        self.decs = nn.ModuleList()
        rev = list(reversed(chs))
        for k, c in enumerate(rev):
            nxt = rev[k + 1] if k + 1 < len(rev) else chs[0]
            self.decs.append(LinkDec(c, nxt))
        self.head = nn.Sequential(cbr(chs[0], chs[0]), nn.Conv1d(chs[0], out_ch, 1))
    def forward(self, x):
        h = self.stem(x)
        skips = []
        for e in self.encs:
            h = e(h); skips.append(h)
        h = self.bott(h)
        for k, d in enumerate(self.decs):
            h = d(h)
            j = len(skips) - 2 - k
            if j >= 0:
                s = skips[j]
                if h.shape[-1] != s.shape[-1]:
                    h = F.interpolate(h, size=s.shape[-1], mode="linear", align_corners=False)
                h = h + s
        return {"wave": torch.tanh(self.head(h))}

# ------------------------------------------------------------------ FPN
class FPN1D(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=64, levels=4, pyr=128):
        super().__init__()
        chs = [base * (2 ** i) for i in range(levels)]
        self.stem = cbr(in_ch, chs[0], 7, 1)
        self.encs = nn.ModuleList()
        prev = chs[0]
        for c in chs:
            self.encs.append(LinkEnc(prev, c, 2)); prev = c
        self.lat = nn.ModuleList([nn.Conv1d(c, pyr, 1) for c in chs])
        self.smooth = nn.ModuleList([cbr(pyr, pyr) for _ in chs])
        self.heads = nn.ModuleList([nn.Sequential(cbr(pyr, pyr // 2), cbr(pyr // 2, pyr // 2))
                                    for _ in chs])
        self.head = nn.Sequential(cbr(pyr // 2, pyr // 2), nn.Conv1d(pyr // 2, out_ch, 1))
    def forward(self, x):
        L = x.shape[-1]
        h = self.stem(x); feats = []
        for e in self.encs:
            h = e(h); feats.append(h)
        ps = [None] * len(feats)
        ps[-1] = self.lat[-1](feats[-1])
        for i in range(len(feats) - 2, -1, -1):
            up = F.interpolate(ps[i + 1], size=feats[i].shape[-1], mode="linear",
                               align_corners=False)
            ps[i] = self.lat[i](feats[i]) + up
        ps = [s(p) for s, p in zip(self.smooth, ps)]
        acc = None
        for hd, p in zip(self.heads, ps):
            v = F.interpolate(hd(p), size=L, mode="linear", align_corners=False)
            acc = v if acc is None else acc + v
        return {"wave": torch.tanh(self.head(acc))}

# ------------------------------------------------------------------ MultiResLinkNet
class MultiResBlock(nn.Module):
    # MultiResUNet block (Ibtehaz & Rahman) in 1-D: three successive 3-conv stages of
    # increasing width, concatenated, plus a 1x1 residual shortcut.
    def __init__(self, cin, U, alpha=1.67):
        super().__init__()
        W = alpha * U
        # max(1, ...): below U=4 the 0.167 stage floors to zero channels, and the failure
        # then surfaces as an opaque conv error rather than pointing here.
        c1, c2, c3 = (max(1, int(W * 0.167)), max(1, int(W * 0.333)), max(1, int(W * 0.5)))
        self.out_channels = c1 + c2 + c3
        self.sc = nn.Sequential(nn.Conv1d(cin, self.out_channels, 1, bias=False),
                                nn.BatchNorm1d(self.out_channels))
        self.a = cbr(cin, c1); self.b = cbr(c1, c2); self.c = cbr(c2, c3)
        self.bn1 = nn.BatchNorm1d(self.out_channels)
        self.bn2 = nn.BatchNorm1d(self.out_channels)
    def forward(self, x):
        s = self.sc(x)
        a = self.a(x); b = self.b(a); c = self.c(b)
        o = self.bn1(torch.cat([a, b, c], 1))
        return F.relu(self.bn2(o + s))

class ResPath(nn.Module):
    # Processes an encoder feature before it is added to the decoder, instead of a raw skip.
    def __init__(self, ch, length):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(max(1, length)):
            self.blocks.append(nn.ModuleDict({
                "sc": nn.Sequential(nn.Conv1d(ch, ch, 1, bias=False), nn.BatchNorm1d(ch)),
                "cv": nn.Sequential(nn.Conv1d(ch, ch, 3, padding=1, bias=False),
                                    nn.BatchNorm1d(ch)),
            }))
    def forward(self, x):
        for b in self.blocks:
            x = F.relu(b["sc"](x) + b["cv"](x))
        return x

class MultiResLinkNet1D(nn.Module):
    # LinkNet skeleton, MultiRes blocks instead of plain convolutions, ResPath skips added
    # (not concatenated), and deep supervision from every encoder level.
    def __init__(self, in_ch=1, out_ch=1, base=64, levels=4, deep_supervision=True):
        super().__init__()
        self.deep_supervision = deep_supervision
        units = [base * (2 ** i) for i in range(levels)]
        self.stem = cbr(in_ch, base, 7, 1)
        self.encs = nn.ModuleList(); self.paths = nn.ModuleList()
        prev = base; enc_ch = []
        for i, u in enumerate(units):
            blk = MultiResBlock(prev, u)
            self.encs.append(blk)
            self.paths.append(ResPath(blk.out_channels, levels - i))
            enc_ch.append(blk.out_channels); prev = blk.out_channels
        self.bott = MultiResBlock(prev, units[-1])
        self.decs = nn.ModuleList()
        rev_ch = list(reversed(enc_ch))
        cur = self.bott.out_channels
        # Decoder step k must emerge with the channel count AND length of skips[-1-k], or
        # the additive skip is silently dropped and every ResPath receives zero gradient.
        # Encoder here pools AFTER appending the skip, so the target is rev_ch[k] -- not
        # rev_ch[k+1], which is correct only for the stride-2 encoder in LinkNet1D.
        for k in range(levels):
            tgt = rev_ch[k]
            self.decs.append(LinkDec(cur, tgt)); cur = tgt
        self.head = nn.Sequential(cbr(cur, base), nn.Conv1d(base, out_ch, 1))
        self.aux = nn.ModuleList([nn.Conv1d(c, out_ch, 1) for c in enc_ch]) \
                   if deep_supervision else None
    def forward(self, x):
        L = x.shape[-1]
        h = self.stem(x)
        skips = []
        for e in self.encs:
            h = e(h); skips.append(h)
            h = F.max_pool1d(h, 2)
        h = self.bott(h)
        for k, d in enumerate(self.decs):
            h = d(h)
            j = len(skips) - 1 - k
            if j >= 0:
                s = self.paths[j](skips[j])
                if h.shape[-1] != s.shape[-1]:
                    h = F.interpolate(h, size=s.shape[-1], mode="linear", align_corners=False)
                if h.shape[1] != s.shape[1]:
                    raise RuntimeError(                     # raise, not assert: an invariant
                        f"skip channel mismatch at decoder {k}: {h.shape[1]} vs "        # this
                        f"{s.shape[1]}. Dropping it silently is what cost 59% of this "  # load
                        "model's gradient once already.")   # bearing must survive python -O
                h = h + s
        if h.shape[-1] != L:
            h = F.interpolate(h, size=L, mode="linear", align_corners=False)
        out = {"wave": torch.tanh(self.head(h))}
        if self.aux is not None and self.training:
            out["aux"] = [F.interpolate(a(s), size=L, mode="linear", align_corners=False)
                          for a, s in zip(self.aux, skips)]
        return out

BASELINES = {"fpn": FPN1D, "unet": UNet1D, "linknet": LinkNet1D,
             "multireslinknet": MultiResLinkNet1D}

def build_baseline(name, in_ch=1, out_ch=1, base=64, levels=4):
    return BASELINES[name](in_ch=in_ch, out_ch=out_ch, base=base, levels=levels)

def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
'''

# --------------------------------------------------------------------------- cardiomamba
CRVS_CMNET_SRC = r'''
# crvs_cmnet.py -- CardioMamba-Net (contributions C1-C5 of PLAN.md).
# C2 dual-domain encoder, C3 bidirectional SSM bottleneck, C4 multi-task decoder with
# peak-conditioned FiLM refinement. C1 lives in the data pipeline, C5 in crvs_losses.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from crvs_models import cbr, MultiResBlock, ResPath, LinkDec, count_params

class ECA(nn.Module):
    # Efficient channel attention: a length-k 1-D conv over the channel descriptor.
    def __init__(self, ch, k=5):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, k, padding=k // 2, bias=False)
    def forward(self, x):
        w = x.mean(-1, keepdim=True).transpose(1, 2)
        w = torch.sigmoid(self.conv(w)).transpose(1, 2)
        return x * w

class LiftingUnit(nn.Module):
    # Learnable second-generation wavelet: split into even/odd, predict, update.
    # Replaces a fixed wavelet basis with one the network chooses for radar.
    def __init__(self, ch, k=5):
        super().__init__()
        self.P = nn.Sequential(nn.Conv1d(ch, ch, k, padding=k // 2), nn.Tanh(),
                               nn.Conv1d(ch, ch, 1))
        self.U = nn.Sequential(nn.Conv1d(ch, ch, k, padding=k // 2), nn.Tanh(),
                               nn.Conv1d(ch, ch, 1))
    def forward(self, x):
        xe, xo = x[..., ::2], x[..., 1::2]
        n = min(xe.shape[-1], xo.shape[-1])
        xe, xo = xe[..., :n], xo[..., :n]
        d = xo - self.P(xe)
        c = xe + self.U(d)
        return c, d

class WaveletBranch(nn.Module):
    # Multi-resolution analysis producing one feature map per scale, to sit alongside the
    # convolutional branch. LifWavNet uses this idea as the whole network; here it is half
    # of a dual-domain encoder.
    #
    # Level 0 is taken at the INPUT resolution, before any lifting. Encoder level i sits at
    # L/2^i, and a lifting unit halves length, so starting the branch with a lifting step
    # would put every wavelet feature one octave below its conv counterpart and force a 2x
    # upsample at every fusion. The dual-domain claim (C2) is that the two branches see the
    # SAME scale from different domains, so they have to be aligned octave for octave.
    def __init__(self, ch, out_chs, levels=4):
        super().__init__()
        self.proj0 = nn.Conv1d(ch, out_chs[0], 1)
        self.units = nn.ModuleList([LiftingUnit(ch) for _ in range(max(levels - 1, 0))])
        self.proj = nn.ModuleList([nn.Conv1d(ch * 2, o, 1) for o in out_chs[1:]])
    def forward(self, x):
        feats = [self.proj0(x)]
        c = x
        for u, p in zip(self.units, self.proj):
            c, d = u(c)
            feats.append(p(torch.cat([c, d], 1)))
        return feats

class S4D(nn.Module):
    # Diagonal state-space layer (S4D-Lin). Pure PyTorch: an FFT convolution with a kernel
    # built from learned diagonal dynamics. No custom CUDA, so it always builds on Kaggle.
    def __init__(self, d_model, d_state=64, dt_min=1e-3, dt_max=1e-1):
        super().__init__()
        H, N = d_model, d_state // 2
        log_dt = torch.rand(H) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)
        self.log_A_real = nn.Parameter(torch.log(0.5 * torch.ones(H, N)))
        self.A_imag = nn.Parameter(math.pi * torch.arange(N).float().repeat(H, 1))
        self.C = nn.Parameter(torch.randn(H, N, 2) * (0.5 ** 0.5))
        self.D = nn.Parameter(torch.randn(H))
    def kernel(self, L, device, dtype=torch.float32):
        dt = torch.exp(self.log_dt).to(dtype).unsqueeze(-1)
        A = -torch.exp(self.log_A_real.to(dtype)) + 1j * self.A_imag.to(dtype)
        C = torch.view_as_complex(self.C.to(dtype).contiguous())
        dtA = A * dt
        n = torch.arange(L, device=device, dtype=dtype)
        K = dtA.unsqueeze(-1) * n
        Cc = C * (torch.exp(dtA) - 1.0) / A
        return 2.0 * torch.einsum("hn,hnl->hl", Cc, torch.exp(K)).real
    def forward(self, u):
        L = u.shape[-1]
        uf = u.float()
        k = self.kernel(L, u.device)
        n = 2 * L
        y = torch.fft.irfft(torch.fft.rfft(uf, n=n) * torch.fft.rfft(k, n=n), n=n)[..., :L]
        y = y + uf * self.D.unsqueeze(-1)
        return y.to(u.dtype)

class BiSSM(nn.Module):
    # Bidirectional SSM block: an 8 s window holds 8-10 cardiac cycles, and this is what
    # lets beat n inform beat n+1. Linear time in sequence length.
    def __init__(self, d, d_state=64, expand=2, dropout=0.1):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.fwd = S4D(d, d_state)
        self.bwd = S4D(d, d_state)
        self.mix = nn.Conv1d(2 * d, d, 1)
        self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Conv1d(d, expand * d, 1), nn.GELU(),
                                nn.Dropout(dropout), nn.Conv1d(expand * d, d, 1))
    def forward(self, x):
        h = self.n1(x.transpose(1, 2)).transpose(1, 2)
        f = self.fwd(h)
        b = self.bwd(h.flip(-1)).flip(-1)
        x = x + self.mix(torch.cat([f, b], 1))
        h = self.n2(x.transpose(1, 2)).transpose(1, 2)
        return x + self.ff(h)

class TransformerBottleneck(nn.Module):
    # The fair-fight control for C3: same budget, attention instead of an SSM.
    def __init__(self, d, nhead=8, layers=3, dropout=0.1, max_len=1024):
        super().__init__()
        self.pos = nn.Parameter(torch.randn(1, max_len, d) * 0.02)
        lyr = nn.TransformerEncoderLayer(d, nhead, dim_feedforward=2 * d, dropout=dropout,
                                         batch_first=True, norm_first=True,
                                         activation="gelu")
        self.enc = nn.TransformerEncoder(lyr, layers)
    def forward(self, x):
        h = x.transpose(1, 2)
        h = h + self.pos[:, :h.shape[1]]
        return self.enc(h).transpose(1, 2)

class FiLM(nn.Module):
    # Peak-conditioned refinement: the R-peak head tells the waveform head where a QRS
    # belongs BEFORE it draws one. Modulation is per-sample, because a QRS is localised.
    def __init__(self, cond_ch, feat_ch, k=9):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(cond_ch, feat_ch, k, padding=k // 2), nn.GELU(),
                                 nn.Conv1d(feat_ch, 2 * feat_ch, 1))
    def forward(self, feat, cond):
        g, b = self.net(cond).chunk(2, 1)
        return feat * (1.0 + torch.tanh(g)) + b

class CardioMambaNet(nn.Module):
    def __init__(self, in_ch=8, base=32, levels=4, d_ssm=256, ssm_blocks=3, d_state=64,
                 bottleneck="ssm", use_wavelet=True, multitask=True, use_film=True,
                 dropout=0.1):
        super().__init__()
        self.use_wavelet = use_wavelet
        self.multitask = multitask
        self.use_film = use_film and multitask
        units = [base * (2 ** i) for i in range(levels)]
        # C1 lands here: a learnable 1x1 mix over the 8 physics channels, so the network
        # can rediscover arctangent demodulation if that really is optimal.
        self.mix = nn.Sequential(nn.Conv1d(in_ch, 32, 1), nn.GELU())
        self.stem = cbr(32, base, 7, 1)
        self.encs = nn.ModuleList(); self.paths = nn.ModuleList()
        prev = base; enc_ch = []
        for i, u in enumerate(units):
            blk = MultiResBlock(prev, u)
            self.encs.append(blk)
            self.paths.append(ResPath(blk.out_channels, levels - i))
            enc_ch.append(blk.out_channels); prev = blk.out_channels
        if use_wavelet:
            self.wave = WaveletBranch(base, enc_ch, levels)
            self.fuse = nn.ModuleList([nn.Sequential(nn.Conv1d(2 * c, c, 1), ECA(c))
                                       for c in enc_ch])
        self.pre = nn.Conv1d(prev, d_ssm, 1)
        if bottleneck == "ssm":
            self.bott = nn.Sequential(*[BiSSM(d_ssm, d_state, dropout=dropout)
                                        for _ in range(ssm_blocks)])
        elif bottleneck == "transformer":
            self.bott = TransformerBottleneck(d_ssm, layers=ssm_blocks, dropout=dropout)
        else:
            self.bott = nn.Sequential(cbr(d_ssm, d_ssm), cbr(d_ssm, d_ssm))
        self.post = nn.Conv1d(d_ssm, prev, 1)
        self.decs = nn.ModuleList()
        rev = list(reversed(enc_ch)); cur = prev
        # Same indexing rule as MultiResLinkNet1D: target rev[k], so decoder step k lines up
        # with skips[-1-k] in both channels and length and the ResPath actually contributes.
        for k in range(levels):
            tgt = rev[k]
            self.decs.append(LinkDec(cur, tgt)); cur = tgt
        self.refine = cbr(cur, base)
        self.head_wave = nn.Conv1d(base, 1, 1)
        if multitask:
            self.head_peak = nn.Sequential(cbr(cur, base), nn.Conv1d(base, 1, 1))
            rr_ch = max(base // 2, 4)
            self.head_rr = nn.Sequential(cbr(cur, rr_ch), nn.Conv1d(rr_ch, 1, 1))
            if self.use_film:
                self.film = FiLM(1, base)
    def forward(self, x):
        L = x.shape[-1]
        h = self.stem(self.mix(x))
        wfeat = self.wave(h) if self.use_wavelet else None
        skips = []
        for i, e in enumerate(self.encs):
            h = e(h)
            if wfeat is not None:
                w = wfeat[i]
                if w.shape[-1] != h.shape[-1]:
                    w = F.interpolate(w, size=h.shape[-1], mode="linear", align_corners=False)
                h = self.fuse[i](torch.cat([h, w], 1))
            skips.append(h)
            h = F.max_pool1d(h, 2)
        h = self.post(self.bott(self.pre(h)))
        for k, d in enumerate(self.decs):
            h = d(h)
            j = len(skips) - 1 - k
            if j >= 0:
                s = self.paths[j](skips[j])
                if h.shape[-1] != s.shape[-1]:
                    h = F.interpolate(h, size=s.shape[-1], mode="linear", align_corners=False)
                if h.shape[1] != s.shape[1]:
                    raise RuntimeError(
                        f"skip channel mismatch at decoder {k}: {h.shape[1]} vs {s.shape[1]}")
                h = h + s
        if h.shape[-1] != L:
            h = F.interpolate(h, size=L, mode="linear", align_corners=False)
        out = {}
        if self.multitask:
            peak_logit = self.head_peak(h)
            out["peak"] = peak_logit
            out["rr"] = F.softplus(self.head_rr(h))
            f = self.refine(h)
            if self.use_film:
                # Gradient flows through the conditioning on purpose: the peak head is meant
                # to be shaped by the waveform loss as well as its own, which is the point of
                # peak-conditioned refinement. Detaching here would make it a one-way hint.
                f = self.film(f, torch.sigmoid(peak_logit))
            out["wave"] = torch.tanh(self.head_wave(f))
        else:
            out["wave"] = torch.tanh(self.head_wave(self.refine(h)))
        return out

def build_cmnet(**kw):
    return CardioMambaNet(**kw)
'''

# --------------------------------------------------------------------------- losses
CRVS_LOSS_SRC = r'''
# crvs_losses.py -- C5, the morphology-aware composite loss.
# Plain MSE is the conditional mean, so it flattens the R peak; that is exactly why the
# baseline over-estimates RMSSD by ~2x. Every term here exists to stop that.
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiResSTFTLoss(nn.Module):
    # Spectral convergence + log-magnitude at three resolutions. Forces the model to get
    # the spectrum right, not just the sample-wise average.
    def __init__(self, ffts=(256, 128, 64)):
        super().__init__()
        self.ffts = ffts
    def _one(self, y, yh, n):
        hop, win = n // 4, n
        w = torch.hann_window(win, device=y.device, dtype=torch.float32)
        kw = dict(n_fft=n, hop_length=hop, win_length=win, window=w,
                  return_complex=True, center=True, pad_mode="reflect")
        Y = torch.stft(y, **kw).abs().clamp_min(1e-7)
        H = torch.stft(yh, **kw).abs().clamp_min(1e-7)
        sc = torch.norm(Y - H, p="fro", dim=(-2, -1)) / (torch.norm(Y, p="fro", dim=(-2, -1)) + 1e-7)
        mag = F.l1_loss(torch.log(H), torch.log(Y))
        return sc.mean() + mag
    def forward(self, y, yh):
        y = y.squeeze(1).float(); yh = yh.squeeze(1).float()
        return sum(self._one(y, yh, n) for n in self.ffts) / len(self.ffts)

def pearson_loss(y, yh, eps=1e-8):
    y = y.squeeze(1).float(); yh = yh.squeeze(1).float()
    y = y - y.mean(-1, keepdim=True); yh = yh - yh.mean(-1, keepdim=True)
    num = (y * yh).sum(-1)
    den = y.norm(dim=-1) * yh.norm(dim=-1) + eps
    return (1.0 - num / den).mean()

def focal_bce(logit, target, alpha=0.75, gamma=2.0):
    p = torch.sigmoid(logit)
    ce = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
    pt = p * target + (1 - p) * (1 - target)
    w = alpha * target + (1 - alpha) * (1 - target)
    return (w * (1 - pt).pow(gamma) * ce).mean()

class CompositeLoss(nn.Module):
    def __init__(self, w_huber=1.0, w_stft=0.5, w_peak=0.3, w_rr=0.1,
                 w_peakw=0.5, w_corr=0.3, huber_delta=0.1, peak_weight=4.0):
        super().__init__()
        self.w = dict(huber=w_huber, stft=w_stft, peak=w_peak, rr=w_rr,
                      peakw=w_peakw, corr=w_corr)
        self.delta = huber_delta
        self.peak_weight = peak_weight
        self.stft = MultiResSTFTLoss()
    def forward(self, pred, y, pk=None, rr=None):
        parts = {}
        wave = pred["wave"]
        if self.w["huber"]:
            parts["huber"] = F.huber_loss(wave, y, delta=self.delta)
        if self.w["stft"]:
            parts["stft"] = self.stft(y, wave)
        if self.w["corr"]:
            parts["corr"] = pearson_loss(y, wave)
        if self.w["peakw"] and pk is not None:
            wgt = 1.0 + self.peak_weight * pk
            parts["peakw"] = ((wgt * (wave - y).abs()).sum() / (wgt.sum() + 1e-8))
        if self.w["peak"] and pk is not None and "peak" in pred:
            parts["peak"] = focal_bce(pred["peak"], pk)
        if self.w["rr"] and rr is not None and "rr" in pred:
            parts["rr"] = F.l1_loss(pred["rr"], rr)
        if "aux" in pred:
            parts["aux"] = sum(F.huber_loss(a, y, delta=self.delta)
                               for a in pred["aux"]) / max(len(pred["aux"]), 1) * 0.2
        total = sum(self.w.get(k, 1.0) * v for k, v in parts.items())
        return total, {k: float(v.detach()) for k, v in parts.items()}

class MSEOnly(nn.Module):
    # The baseline's objective, kept verbatim so ablation row 1 is a true reproduction.
    def forward(self, pred, y, pk=None, rr=None):
        l = F.mse_loss(pred["wave"], y)
        if "aux" in pred:
            l = l + 0.2 * sum(F.mse_loss(a, y) for a in pred["aux"]) / max(len(pred["aux"]), 1)
        return l, {"mse": float(l.detach())}
'''

# --------------------------------------------------------------------------- engine
CRVS_ENGINE_SRC = r'''
# crvs_engine.py -- deterministic dual-GPU engine with mid-epoch recovery and telemetry.
import csv, json, math, time, os, random, shutil, subprocess, threading, hashlib
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ENGINE_VERSION = 7

def _autocast(device_type, enabled):
    try:
        return torch.amp.autocast(device_type=device_type, enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(enabled=enabled)

def _grad_scaler(device_type, enabled):
    try:
        return torch.amp.GradScaler(device_type, enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)

def pick_device():
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        names = [torch.cuda.get_device_name(i) for i in range(n)]
        return torch.device("cuda"), n, names
    return torch.device("cpu"), 0, []

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def _atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, default=str, allow_nan=True)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

def _jsonable(v):
    if isinstance(v, (np.floating, np.integer)): return v.item()
    if isinstance(v, np.ndarray): return v.tolist()
    if isinstance(v, float) and not math.isfinite(v): return None
    return v

def _append_jsonl(path, record):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    clean = {str(k): _jsonable(v) for k, v in record.items()}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(clean, default=str, allow_nan=False) + "\n"); f.flush()

def _mean_parts(sums, n, prefix):
    return {f"{prefix}_{k}": float(v) / max(int(n), 1) for k, v in sums.items()}

def _system_stats(out_dir):
    d = shutil.disk_usage(Path(out_dir))
    rec = {"disk_free_gb": d.free / 2**30, "disk_used_gb": d.used / 2**30}
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rec["process_peak_rss_gb"] = rss / 2**20  # Linux ru_maxrss is KiB
    except Exception:
        pass
    try:
        load = os.getloadavg(); rec.update(cpu_load_1m=load[0], cpu_load_5m=load[1], cpu_load_15m=load[2])
    except Exception:
        pass
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            rec[f"gpu{i}_peak_alloc_gb"] = torch.cuda.max_memory_allocated(i) / 2**30
            rec[f"gpu{i}_peak_reserved_gb"] = torch.cuda.max_memory_reserved(i) / 2**30
        try:
            q = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,temperature.gpu,power.draw,memory.used",
                                "--format=csv,noheader,nounits"], capture_output=True, text=True,
                               timeout=10, check=False)
            for line in q.stdout.strip().splitlines():
                vals = [x.strip() for x in line.split(",")]
                if len(vals) == 5:
                    i = vals[0]
                    for key, val in zip(("util_pct", "temp_c", "power_w", "mem_used_mb"), vals[1:]):
                        try: rec[f"gpu{i}_{key}"] = float(val)
                        except ValueError: pass
        except Exception:
            pass
    return rec

class Trainer:
    def __init__(self, model, loss_fn, out_dir, run_id, sync=None, lr=5e-4, weight_decay=1e-4,
                 epochs=120, patience=20, batch_size=64, num_workers=2, amp=True,
                 multi_gpu=True, grad_clip=1.0, min_lr=1e-6, log_every=25,
                 checkpoint_every_steps=50, checkpoint_every_s=300, seed=42,
                 require_dual_gpu=False, run_config=None, monitor="val_total",
                 monitor_mode="min", min_epochs=0, recovery_lr_factor=0.25,
                 max_numerical_recoveries=3, disable_amp_after_recoveries=2,
                 max_nonfinite_grad_batches=8):
        self.device, self.ngpu, self.gpu_names = pick_device()
        if require_dual_gpu and self.ngpu < 2:
            raise RuntimeError("This training notebook requires Kaggle GPU T4 x2. "
                               "Choose Settings > Accelerator > GPU T4 x2, then restart.")
        self.raw_model = model.to(self.device); self.model = self.raw_model
        if multi_gpu and self.ngpu > 1:
            self.model = nn.DataParallel(self.raw_model)
        self.loss_fn = loss_fn
        self.out = Path(out_dir); self.out.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id; self.sync = sync
        self.epochs = int(epochs); self.patience = int(patience)
        self.monitor = str(monitor)
        self.monitor_mode = str(monitor_mode).lower()
        if self.monitor_mode not in ("min", "max"):
            raise ValueError("monitor_mode must be 'min' or 'max'")
        self.min_epochs = max(0, int(min_epochs))
        self.recovery_lr_factor = float(recovery_lr_factor)
        if not 0.0 < self.recovery_lr_factor < 1.0:
            raise ValueError("recovery_lr_factor must be between 0 and 1")
        self.max_numerical_recoveries = max(1, int(max_numerical_recoveries))
        self.disable_amp_after_recoveries = max(1, int(disable_amp_after_recoveries))
        self.max_nonfinite_grad_batches = max(1, int(max_nonfinite_grad_batches))
        self.initial_lr = float(lr)
        self.bs = int(batch_size); self.nw = int(num_workers); self.seed = int(seed)
        self.amp = bool(amp and self.device.type == "cuda"); self.grad_clip = grad_clip
        self.opt = torch.optim.AdamW(self.raw_model.parameters(), lr=lr, weight_decay=weight_decay)
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=self.epochs, eta_min=min_lr)
        self.scaler = _grad_scaler(self.device.type, self.amp)
        self.log_every = max(1, int(log_every))
        self.checkpoint_every_steps = max(1, int(checkpoint_every_steps))
        self.checkpoint_every_s = max(30, int(checkpoint_every_s))
        self.config = dict(run_config or {})
        self.config_hash = hashlib.sha256(json.dumps(
            self.config, sort_keys=True, default=str).encode()).hexdigest()
        self.state = {"schema_version": 5, "engine_version": ENGINE_VERSION,
                      "epoch": 0, "active_epoch": 0, "batch_in_epoch": 0,
                      "global_step": 0,
                      "best": float("inf") if self.monitor_mode == "min" else -float("inf"),
                      "best_epoch": -1, "monitor": self.monitor,
                      "monitor_mode": self.monitor_mode, "min_epochs": self.min_epochs,
                      "bad_epochs": 0, "history": [], "partial": {},
                      "run_id": run_id, "done": False, "config_hash": self.config_hash,
                      "created_utc": datetime.now(timezone.utc).isoformat()}
        self._save_lock = threading.Lock(); self._last_checkpoint = time.time()
        self._active = False
        _atomic_json(self.out / "run_config.json", self.config)
        _atomic_json(self.out / "environment.json", {
            "engine_version": ENGINE_VERSION, "torch": torch.__version__,
            "cuda": torch.version.cuda, "gpu_count": self.ngpu, "gpu_names": self.gpu_names,
            "amp": self.amp, "python": os.sys.version, "config_hash": self.config_hash,
            "recovery_lr_factor": self.recovery_lr_factor,
            "max_numerical_recoveries": self.max_numerical_recoveries,
            "disable_amp_after_recoveries": self.disable_amp_after_recoveries,
            "max_nonfinite_grad_batches": self.max_nonfinite_grad_batches})

    @property
    def ckpt(self):
        return self.out / "state.pt"

    def _payload(self):
        return {"model": self.raw_model.state_dict(), "opt": self.opt.state_dict(),
                "sched": self.sched.state_dict(), "scaler": self.scaler.state_dict(),
                "amp_enabled": bool(self.amp),
                "state": self.state, "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                "np_rng": np.random.get_state(), "python_rng": random.getstate(),
                "config": self.config, "config_hash": self.config_hash,
                "saved_utc": datetime.now(timezone.utc).isoformat()}

    def save(self, tag="state", reason="checkpoint"):
        with self._save_lock:
            path = self.out / (tag + ".pt"); tmp = path.with_suffix(".pt.tmp")
            torch.save(self._payload(), tmp); os.replace(tmp, path)
            _atomic_json(self.out / "state.json", self.state)
            self._last_checkpoint = time.time()
        if self.sync:
            self.sync.mark_dirty(f"{self.run_id}:{reason}")

    def emergency_checkpoint(self):
        if self._active:
            self.save("state", reason="interrupt-emergency")

    @staticmethod
    def _resume_config_view(config):
        # Library hashes and the recovery policy may change only to repair the runtime. Model,
        # data, split, loss and optimizer semantics must remain byte-for-byte compatible.
        clean = dict(config or {})
        clean.pop("library_sha256", None)
        clean.pop("recovery_policy", None)
        return json.dumps(clean, sort_keys=True, default=str)

    def _check_resume_config(self, payload):
        got_hash = payload.get("config_hash", payload.get("state", {}).get("config_hash"))
        if not got_hash or got_hash == self.config_hash:
            return
        old = self._resume_config_view(payload.get("config", {}))
        new = self._resume_config_view(self.config)
        if old != new:
            raise RuntimeError("checkpoint training configuration differs from this run. "
                               "Use a new RUN_ID or restore the original configuration.")
        print("  compatible engine/recovery-policy upgrade detected; preserving the run")

    @staticmethod
    def _model_payload_is_finite(payload):
        for value in payload.get("model", {}).values():
            if torch.is_tensor(value) and not bool(torch.isfinite(value).all()):
                return False
        return True

    def _restore_payload(self, payload):
        self.raw_model.load_state_dict(payload["model"], strict=True)
        self.opt.load_state_dict(payload["opt"])
        self.sched.load_state_dict(payload["sched"])
        # A disabled GradScaler intentionally serializes to {}. Older engine-v6
        # checkpoints did not persist the AMP flag, so loading that empty state into
        # a newly enabled scaler raises "source state dict is empty" before the run
        # can resume. Treat an old empty scaler as proof that AMP had been disabled;
        # new checkpoints also carry the explicit flag.
        scaler_state = payload.get("scaler") or {}
        saved_amp = payload.get("amp_enabled")
        if saved_amp is None:
            saved_amp = bool(scaler_state)
        self.amp = bool(saved_amp and self.device.type == "cuda")
        self.scaler = _grad_scaler(self.device.type, self.amp)
        if self.amp and scaler_state:
            self.scaler.load_state_dict(scaler_state)
        self.state = payload["state"]
        if payload.get("torch_rng") is not None:
            torch.set_rng_state(payload["torch_rng"].cpu())
        if payload.get("np_rng") is not None:
            np.random.set_state(payload["np_rng"])
        if payload.get("python_rng") is not None:
            random.setstate(payload["python_rng"])
        if torch.cuda.is_available() and payload.get("cuda_rng"):
            torch.cuda.set_rng_state_all([x.cpu() for x in payload["cuda_rng"]])

    def load(self):
        if not self.ckpt.exists():
            return False
        try:
            current = torch.load(self.ckpt, map_location=self.device, weights_only=False)
            self._check_resume_config(current)
            failed_state = dict(current.get("state", {}))
            last_error = str(failed_state.get("last_error", ""))
            numerical_error = ("non-finite" in last_error.lower() or
                               "floatingpointerror" in last_error.lower())
            poisoned_model = not self._model_payload_is_finite(current)
            recovery_needed = numerical_error or poisoned_model
            if recovery_needed:
                prior_count = int(failed_state.get("recovery_count", 0) or 0)
                if prior_count >= self.max_numerical_recoveries:
                    raise RuntimeError(
                        f"numerical recovery limit ({self.max_numerical_recoveries}) reached; "
                        "do not silently finalize this run")
                best_path = self.out / "best.pt"
                if not best_path.exists():
                    raise RuntimeError("numerical checkpoint failure and best.pt is unavailable")
                chosen = torch.load(best_path, map_location=self.device, weights_only=False)
                self._check_resume_config(chosen)
                if not self._model_payload_is_finite(chosen):
                    raise RuntimeError("best.pt also contains non-finite model parameters")
                self._restore_payload(chosen)
                recovery_count = prior_count + 1
                eta_min = float(getattr(self.sched, "eta_min", 1e-6))
                target_lr = max(eta_min, self.initial_lr *
                                (self.recovery_lr_factor ** recovery_count))
                for group in self.opt.param_groups:
                    group["lr"] = min(float(group["lr"]), target_lr)
                if hasattr(self.sched, "base_lrs"):
                    self.sched.base_lrs = [min(float(v), target_lr)
                                           for v in self.sched.base_lrs]
                if hasattr(self.sched, "_last_lr"):
                    self.sched._last_lr = [float(g["lr"]) for g in self.opt.param_groups]
                amp_disabled_by_policy = recovery_count >= self.disable_amp_after_recoveries
                if amp_disabled_by_policy:
                    self.amp = False
                    self.scaler = _grad_scaler(self.device.type, False)
                event = {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "reason": last_error or "non-finite model parameters",
                    "failed_epoch": failed_state.get("active_epoch", failed_state.get("epoch")),
                    "failed_batch": failed_state.get("batch_in_epoch", 0),
                    "rollback_epoch": int(self.state.get("epoch", 0)),
                    "rollback_best_epoch": int(self.state.get("best_epoch", -1)),
                    "recovery_count": recovery_count,
                    "effective_lr": float(self.opt.param_groups[0]["lr"]),
                    "amp_disabled": bool(not self.amp),
                    "engine_version": ENGINE_VERSION,
                }
                previous_events = list(failed_state.get("recovery_events", []))
                self.state["recovery_events"] = previous_events + [event]
                self.state["recovery_count"] = recovery_count
                self.state["last_recovered_error"] = event["reason"]
                self.state.pop("last_error", None)
                self.state.pop("stop_reason", None)
                self.state.pop("finished_utc", None)
                self.state.update(done=False, active_epoch=int(self.state.get("epoch", 0)),
                                  batch_in_epoch=0, partial={}, engine_version=ENGINE_VERSION,
                                  config_hash=self.config_hash)
                _append_jsonl(self.out / "recovery_events.jsonl", event)
                self.save("state", reason="numerical-rollback-to-best")
                print(f"  RECOVERED {self.run_id}: epoch "
                      f"{event['failed_epoch']} -> best epoch {event['rollback_best_epoch']}; "
                      f"lr={event['effective_lr']:.2e}; AMP={'on' if self.amp else 'off'}")
            else:
                self._restore_payload(current)
                self.state.update(engine_version=ENGINE_VERSION, config_hash=self.config_hash)
                self.state.pop("last_error", None)
                print(f"  resumed {self.run_id}: completed_epoch={self.state['epoch']}, "
                      f"active_epoch={self.state.get('active_epoch')}, "
                      f"completed_batches={self.state.get('batch_in_epoch', 0)}")
            return True
        except Exception as e:
            raise RuntimeError(f"Checkpoint exists but cannot be resumed safely: "
                               f"{type(e).__name__}: {e}") from e

    def _loader(self, ds, shuffle, epoch=0):
        if len(ds) == 0:
            raise RuntimeError("empty dataset -- check the subject split")
        if hasattr(ds, "set_epoch"):
            ds.set_epoch(epoch)
        drop = bool(shuffle) and len(ds) > self.bs
        gen = torch.Generator(); gen.manual_seed(self.seed + int(epoch) * 1000003)
        return DataLoader(ds, batch_size=min(self.bs, max(len(ds), 1)), shuffle=shuffle,
                          generator=gen, num_workers=self.nw,
                          pin_memory=(self.device.type == "cuda"), drop_last=drop,
                          persistent_workers=False)

    def _step(self, batch, train):
        x, y, pk, rr = [b.to(self.device, non_blocking=True) for b in batch]
        with _autocast(self.device.type, self.amp):
            pred = self.model(x)
            if isinstance(pred, dict) and "aux" in pred and not train:
                pred = {k: v for k, v in pred.items() if k != "aux"}
            loss, parts = self.loss_fn(pred, y, pk, rr)
        return loss, {k: float(v) for k, v in parts.items()}, pred, y, pk, rr

    def _write_batch(self, rec):
        _append_jsonl(self.out / "batch_metrics.jsonl", rec)

    def _validation(self, val_ds, epoch):
        self.model.eval(); sums = {}; n = 0; Y = []; P = []; PK = []; PH = []; RR = []; RH = []
        t0 = time.time()
        with torch.no_grad():
            for batch in self._loader(val_ds, False, epoch):
                loss, parts, pred, y, pk, rr = self._step(batch, False)
                parts = {"total": float(loss), **parts}
                for k, v in parts.items(): sums[k] = sums.get(k, 0.0) + float(v)
                n += 1; Y.append(y.squeeze(1).float().cpu().numpy())
                P.append(pred["wave"].squeeze(1).float().cpu().numpy())
                PK.append(pk.squeeze(1).float().cpu().numpy())
                if "peak" in pred: PH.append(pred["peak"].squeeze(1).float().cpu().numpy())
                RR.append(rr.squeeze(1).float().cpu().numpy())
                if "rr" in pred: RH.append(pred["rr"].squeeze(1).float().cpu().numpy())
        if n == 0: raise RuntimeError("validation loader yielded zero batches")
        y = np.concatenate(Y); p = np.concatenate(P); pk = np.concatenate(PK)
        rec = _mean_parts(sums, n, "val")
        rec["val_seconds"] = time.time() - t0; rec["val_windows"] = len(y)
        rec.update(val_true_mean=float(y.mean()), val_true_std=float(y.std()),
                   val_true_min=float(y.min()), val_true_max=float(y.max()),
                   val_pred_mean=float(p.mean()), val_pred_std=float(p.std()),
                   val_pred_min=float(p.min()), val_pred_max=float(p.max()),
                   val_pred_bias=float((p-y).mean()))
        try:
            from crvs_metrics import seg_metrics, peak_detection_scores, detect_r_peaks, hrv_from_peaks
            global_m = seg_metrics(y.reshape(-1), p.reshape(-1))
            rec.update({"val_" + k: v for k, v in global_m.items()})
            # Per-window waveform diagnostics are compact and retained for every epoch.
            den_y = np.sqrt(np.sum((y - y.mean(1, keepdims=True)) ** 2, axis=1))
            den_p = np.sqrt(np.sum((p - p.mean(1, keepdims=True)) ** 2, axis=1))
            cc = np.sum((y-y.mean(1, keepdims=True))*(p-p.mean(1, keepdims=True)), axis=1) / (den_y*den_p+1e-12)
            rec.update(val_window_CC_temporal_mean=float(100 * np.mean(cc)),
                       val_window_CC_temporal_median=float(100 * np.median(cc)),
                       val_window_CC_temporal_std=float(100 * np.std(cc)),
                       val_window_CC_temporal_p05=float(100 * np.percentile(cc, 5)),
                       val_window_MAE_mean=float(np.mean(np.abs(y-p))),
                       val_window_MSE_mean=float(np.mean((y-p)**2)))
            win = {"epoch": np.full(len(y), epoch + 1), "window": np.arange(len(y)),
                   "mae": np.mean(np.abs(y-p), 1), "mse": np.mean((y-p)**2, 1),
                   "cc_temporal": 100*cc}
            try:
                import pandas as pd
                frame = pd.DataFrame(win)
                if hasattr(val_ds, "index") and len(val_ds.index) == len(frame):
                    for col in ("rec_id", "subject", "scenario_canon", "start"):
                        if col in val_ds.index: frame[col] = val_ds.index[col].to_numpy()
                vd = self.out / "validation_windows"; vd.mkdir(exist_ok=True)
                frame.to_parquet(vd / f"epoch_{epoch+1:04d}.parquet", index=False)
            except Exception as e:
                rec["val_window_table_error"] = f"{type(e).__name__}: {e}"
            # Never concatenate different recordings: that fabricates a beat interval at
            # each boundary. Compute peak/HRV metrics per recording, then macro-average.
            record_rows = []
            if hasattr(val_ds, "index") and len(val_ds.index) == len(y):
                ix = val_ds.index.reset_index(drop=True).assign(_row=np.arange(len(y)))
                for rid, grp in ix.groupby("rec_id"):
                    pos = grp.sort_values("start")["_row"].to_numpy()
                    if len(pos) < 2: continue
                    yg = np.concatenate(y[pos]); pg = np.concatenate(p[pos])
                    peak_s = peak_detection_scores(yg, pg)
                    gt_hrv = hrv_from_peaks(detect_r_peaks(yg))
                    pr_hrv = hrv_from_peaks(detect_r_peaks(pg))
                    rr = {"rec_id": rid, "subject": str(grp["subject"].iloc[0]), **peak_s}
                    for k in ("mean_rr_ms", "sd_rr_ms", "mean_hr_bpm", "sd_hr_bpm", "rmssd_ms"):
                        rr[f"true_{k}"] = gt_hrv[k]; rr[f"pred_{k}"] = pr_hrv[k]
                        rr[f"abs_error_{k}"] = abs(pr_hrv[k]-gt_hrv[k])
                    record_rows.append(rr)
            if record_rows:
                import pandas as pd
                rdf = pd.DataFrame(record_rows)
                rd = self.out / "validation_recordings"; rd.mkdir(exist_ok=True)
                rdf.to_parquet(rd / f"epoch_{epoch+1:04d}.parquet", index=False)
                for k in ("TP", "FP", "FN", "precision", "recall", "F1", "accuracy",
                          "timing_err_ms_median", "timing_err_ms_iqr", "missed_rate"):
                    rec[f"val_wave_peak_{k}"] = float(rdf[k].mean())
                for k in ("mean_rr_ms", "sd_rr_ms", "mean_hr_bpm", "sd_hr_bpm", "rmssd_ms"):
                    rec[f"val_hrv_true_{k}"] = float(rdf[f"true_{k}"].mean())
                    rec[f"val_hrv_pred_{k}"] = float(rdf[f"pred_{k}"].mean())
                    rec[f"val_hrv_abs_error_{k}"] = float(rdf[f"abs_error_{k}"].mean())
        except Exception as e:
            rec["val_signal_metrics_error"] = f"{type(e).__name__}: {e}"
        if PH:
            ph = np.concatenate(PH); prob = 1 / (1 + np.exp(-np.clip(ph, -30, 30)))
            truth = pk >= 0.5; guess = prob >= 0.5
            tp = int(np.sum(truth & guess)); fp = int(np.sum(~truth & guess)); fn = int(np.sum(truth & ~guess))
            prec = tp / max(tp+fp, 1); recall = tp / max(tp+fn, 1)
            rec.update(val_peak_head_TP=tp, val_peak_head_FP=fp, val_peak_head_FN=fn,
                       val_peak_head_precision=prec, val_peak_head_recall=recall,
                       val_peak_head_F1=2*prec*recall/max(prec+recall, 1e-12))
        if RH:
            rr = np.concatenate(RR); rh = np.concatenate(RH); mask = rr > 0
            if np.any(mask):
                rec["val_rr_head_mae_ms"] = float(np.mean(np.abs(rr[mask]-rh[mask]))*1000)
                rec["val_rr_head_rmse_ms"] = float(np.sqrt(np.mean((rr[mask]-rh[mask])**2))*1000)
        return rec

    def _write_epoch(self, rec):
        _append_jsonl(self.out / "epoch_metrics.jsonl", rec)
        # CSV is convenient in Kaggle; JSONL remains the lossless schema-of-record.
        rows = self.state["history"]
        keys = sorted({k for r in rows for k in r})
        tmp = self.out / "epoch_metrics.csv.tmp"
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
            for row in rows: w.writerow({k: _jsonable(row.get(k)) for k in keys})
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, self.out / "epoch_metrics.csv")

    def fit(self, train_ds, val_ds):
        start = int(self.state["epoch"])
        if start >= self.epochs:
            print(f"  {self.run_id} already complete at epoch {start}/{self.epochs}")
            return self.state
        if self.state.get("done"):
            self.state["done"] = False
        self._active = True
        if self.sync: self.sync.set_before_final_flush(self.emergency_checkpoint)
        run_t0 = time.time()
        try:
            for ep in range(start, self.epochs):
                if torch.cuda.is_available():
                    for gpu_i in range(torch.cuda.device_count()):
                        torch.cuda.reset_peak_memory_stats(gpu_i)
                tl = self._loader(train_ds, True, ep)
                resume_batch = int(self.state.get("batch_in_epoch", 0)) if int(self.state.get("active_epoch", ep)) == ep else 0
                partial = self.state.get("partial", {}) if resume_batch else {}
                sums = {k: float(v) for k, v in partial.get("sums", {}).items()}
                n = int(partial.get("n", 0)); samples = int(partial.get("samples", 0))
                grad_sum = float(partial.get("grad_sum", 0)); clip_events = int(partial.get("clip_events", 0))
                nonfinite_grad_skips = int(partial.get("nonfinite_grad_skips", 0))
                epoch_t0 = time.time(); data_t = 0.0; compute_t = 0.0; last_end = time.time()
                self.state.update(active_epoch=ep, batch_in_epoch=resume_batch, done=False)
                self.model.train()
                for i, batch in enumerate(tl):
                    data_t += time.time() - last_end
                    if i < resume_batch:
                        last_end = time.time(); continue
                    step_t0 = time.time(); self.opt.zero_grad(set_to_none=True)
                    loss, parts, _, _, _, _ = self._step(batch, True)
                    if not torch.isfinite(loss):
                        self.save("state", reason="non-finite-loss")
                        raise FloatingPointError(f"non-finite loss at epoch {ep+1}, batch {i+1}")
                    self.scaler.scale(loss).backward(); self.scaler.unscale_(self.opt)
                    grad = float(torch.nn.utils.clip_grad_norm_(
                        self.raw_model.parameters(), self.grad_clip or float("inf")))
                    if not math.isfinite(grad):
                        # GradScaler normally skips this update. Make that behavior explicit,
                        # checkpointable and bounded so a bad batch cannot poison model weights.
                        self.opt.zero_grad(set_to_none=True)
                        self.scaler.update()
                        nonfinite_grad_skips += 1
                        self.state["batch_in_epoch"] = i + 1
                        self.state["partial"] = {
                            "sums": sums, "n": n, "samples": samples,
                            "grad_sum": grad_sum, "clip_events": clip_events,
                            "nonfinite_grad_skips": nonfinite_grad_skips,
                        }
                        self._write_batch({
                            "ts": datetime.now(timezone.utc).isoformat(),
                            "run_id": self.run_id, "epoch": ep+1, "batch": i+1,
                            "batches": len(tl), "event": "nonfinite_gradient_skipped",
                            "grad_norm": grad, "lr": self.opt.param_groups[0]["lr"],
                            "amp_scale": float(self.scaler.get_scale()),
                            "skip_count_epoch": nonfinite_grad_skips,
                        })
                        compute_t += time.time() - step_t0
                        if nonfinite_grad_skips >= self.max_nonfinite_grad_batches:
                            raise FloatingPointError(
                                f"{nonfinite_grad_skips} non-finite gradient batches at "
                                f"epoch {ep+1}; rolling back to best checkpoint")
                        last_end = time.time()
                        continue
                    if self.grad_clip and grad > self.grad_clip: clip_events += 1
                    self.scaler.step(self.opt); self.scaler.update()
                    compute_t += time.time() - step_t0
                    values = {"total": float(loss.detach()), **parts}
                    for k, v in values.items(): sums[k] = sums.get(k, 0.0) + float(v)
                    n += 1; samples += int(batch[0].shape[0]); grad_sum += grad
                    self.state["global_step"] = int(self.state.get("global_step", 0)) + 1
                    self.state["batch_in_epoch"] = i + 1
                    self.state["partial"] = {"sums": sums, "n": n, "samples": samples,
                                             "grad_sum": grad_sum, "clip_events": clip_events,
                                             "nonfinite_grad_skips": nonfinite_grad_skips}
                    if (i + 1) % self.log_every == 0 or i + 1 == len(tl):
                        brec = {"ts": datetime.now(timezone.utc).isoformat(), "run_id": self.run_id,
                                "epoch": ep+1, "batch": i+1, "batches": len(tl),
                                "global_step": self.state["global_step"], "loss": float(loss),
                                "grad_norm": grad, "lr": self.opt.param_groups[0]["lr"],
                                "amp_scale": float(self.scaler.get_scale()),
                                "windows_per_s": int(batch[0].shape[0])/max(time.time()-step_t0, 1e-9)}
                        brec.update({"loss_"+k: v for k, v in parts.items()}); self._write_batch(brec)
                    due_step = self.state["global_step"] % self.checkpoint_every_steps == 0
                    due_time = time.time() - self._last_checkpoint >= self.checkpoint_every_s
                    if due_step or due_time:
                        self.save("state", reason="mid-epoch")
                    last_end = time.time()
                if n == 0: raise RuntimeError("training loader yielded zero batches")
                self.sched.step(); val = self._validation(val_ds, ep)
                rec = {"ts": datetime.now(timezone.utc).isoformat(), "run_id": self.run_id,
                       "epoch": ep+1, "epochs_planned": self.epochs,
                       "global_step": self.state["global_step"], "train_batches": n,
                       "train_windows": samples, "train_grad_norm_mean": grad_sum/max(n,1),
                       "train_grad_clip_events": clip_events,
                       "train_nonfinite_grad_skips": nonfinite_grad_skips,
                       "train_data_seconds": data_t, "train_compute_seconds": compute_t,
                       "train_windows_per_s": samples/max(compute_t, 1e-9),
                       "epoch_seconds": time.time()-epoch_t0,
                       "elapsed_seconds": time.time()-run_t0,
                       "lr": self.opt.param_groups[0]["lr"],
                       "amp_scale": float(self.scaler.get_scale())}
                rec.update(_mean_parts(sums, n, "train")); rec.update(val); rec.update(_system_stats(self.out))
                with torch.no_grad():
                    rec["model_parameter_l2"] = math.sqrt(sum(
                        float(torch.sum(p.detach().float() ** 2)) for p in self.raw_model.parameters()))
                va = float(rec["val_total"])
                if self.monitor not in rec:
                    raise KeyError(f"configured monitor '{self.monitor}' is absent from validation metrics")
                monitored = float(rec[self.monitor])
                if not math.isfinite(monitored):
                    raise FloatingPointError(
                        f"non-finite monitor {self.monitor} at epoch {ep+1}: {monitored}")
                best_so_far = float(self.state["best"])
                improved = ((monitored < best_so_far - 1e-6) if self.monitor_mode == "min"
                            else (monitored > best_so_far + 1e-6))
                if improved:
                    self.state["best"] = monitored; self.state["best_epoch"] = ep+1
                    self.state["bad_epochs"] = 0
                else:
                    self.state["bad_epochs"] = int(self.state.get("bad_epochs", 0)) + 1
                rec.update(improved=bool(improved), monitor=self.monitor,
                           monitor_mode=self.monitor_mode, monitor_value=monitored,
                           best_monitor=float(self.state["best"]),
                           best_val=float(self.state["best"]),
                           best_epoch=int(self.state["best_epoch"]),
                           bad_epochs=int(self.state["bad_epochs"]))
                self.state["epoch"] = ep+1; self.state["active_epoch"] = ep+1
                self.state["batch_in_epoch"] = 0; self.state["partial"] = {}
                self.state["history"].append(rec)
                if improved: self.save("best", reason="new-best-local")
                self.save("state", reason="epoch-complete"); self._write_epoch(rec)
                if self.sync:
                    self.sync.log("epoch", run_id=self.run_id, epoch=ep+1,
                                  train_total=rec.get("train_total"), val_total=va,
                                  cc_t=rec.get("val_CC_temporal"), cc_s=rec.get("val_CC_spectral"),
                                  hr_mae_bpm=rec.get("val_hrv_abs_error_mean_hr_bpm"), improved=improved)
                eta = rec["epoch_seconds"] * max(self.epochs-ep-1, 0) / 3600
                print(f"  ep {ep+1:>3}/{self.epochs} train {rec['train_total']:.5f} "
                      f"val {va:.5f} CCt-win {rec.get('val_window_CC_temporal_mean', float('nan')):.1f} "
                      f"CCt-global {rec.get('val_CC_temporal', float('nan')):.1f} "
                      f"CCs {rec.get('val_CC_spectral', float('nan')):.1f} "
                      f"{'*' if improved else ''} {rec['epoch_seconds']:.0f}s ETA {eta:.1f}h")
                if ((ep + 1) >= self.min_epochs and
                        int(self.state["bad_epochs"]) >= self.patience):
                    self.state["stop_reason"] = "early_stopping"; break
            self.state["done"] = True
            self.state["finished_utc"] = datetime.now(timezone.utc).isoformat()
            self.save("state", reason="run-complete")
            if self.sync: self.sync.log("training_complete", run_id=self.run_id)
            return self.state
        except KeyboardInterrupt:
            # SIGINT normally reaches HFSync first: its final hook already saved and pushed.
            # Avoid a duplicate commit; a directly-raised KeyboardInterrupt still takes this path.
            if time.time() - self._last_checkpoint > 2:
                self.save("state", reason="keyboard-interrupt")
            if self.sync:
                if not self.sync.recently_pushed(5):
                    self.sync.flush(final=True, force=True,
                                    msg=f"{self.run_id} stopped by user", run_final_hook=False)
            raise
        except Exception as e:
            self.state["last_error"] = f"{type(e).__name__}: {e}"
            self.save("state", reason="training-error")
            if self.sync:
                self.sync.log("training_error", run_id=self.run_id, error=self.state["last_error"])
                self.sync.flush(final=True, force=True,
                                msg=f"{self.run_id} error checkpoint", run_final_hook=False)
            raise
        finally:
            self._active = False
            if self.sync: self.sync.set_before_final_flush(None)

    @torch.no_grad()
    def predict(self, ds, max_keep=None):
        bp = self.out / "best.pt"
        if bp.exists():
            d = torch.load(bp, map_location=self.device, weights_only=False)
            self.raw_model.load_state_dict(d["model"], strict=True)
        self.model.eval(); Y = []; P = []
        for batch in self._loader(ds, False, 0):
            x, y, pk, rr = [b.to(self.device, non_blocking=True) for b in batch]
            with _autocast(self.device.type, self.amp): out = self.model(x)
            Y.append(y.squeeze(1).float().cpu().numpy())
            P.append(out["wave"].squeeze(1).float().cpu().numpy())
        Y = np.concatenate(Y); P = np.concatenate(P)
        if max_keep is not None: return Y[:max_keep], P[:max_keep]
        return Y, P
'''
