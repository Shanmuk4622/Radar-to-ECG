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
# crvs_engine.py -- the shared GPU training engine.
# Dual T4 via DataParallel, AMP, cosine schedule, early stopping, and a checkpoint written
# EVERY epoch so a killed session costs nothing. Runs are queued and skipped if already done.
import json, math, time, os
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def _autocast(device_type, enabled):
    # torch.cuda.amp.autocast / GradScaler are deprecated and warn on every step in
    # torch >= 2.4. Use the device-typed API where it exists, fall back where it does not.
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
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

class Trainer:
    def __init__(self, model, loss_fn, out_dir, run_id, sync=None, lr=5e-4, weight_decay=1e-4,
                 epochs=120, patience=20, batch_size=64, num_workers=2, amp=True,
                 multi_gpu=True, grad_clip=1.0, min_lr=1e-6, log_every=50):
        self.device, self.ngpu, self.gpu_names = pick_device()
        self.raw_model = model.to(self.device)
        self.model = self.raw_model
        if multi_gpu and self.ngpu > 1:
            self.model = nn.DataParallel(self.raw_model)
        self.loss_fn = loss_fn
        self.out = Path(out_dir); self.out.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id; self.sync = sync
        self.epochs = epochs; self.patience = patience
        self.bs = batch_size; self.nw = num_workers
        self.amp = amp and self.device.type == "cuda"
        self.grad_clip = grad_clip
        self.opt = torch.optim.AdamW(self.raw_model.parameters(), lr=lr,
                                     weight_decay=weight_decay)
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=epochs,
                                                                eta_min=min_lr)
        self.scaler = _grad_scaler(self.device.type, self.amp)
        self.log_every = log_every
        self.state = {"epoch": 0, "best": float("inf"), "best_epoch": -1, "history": [],
                      "run_id": run_id, "done": False}

    @property
    def ckpt(self):
        return self.out / "state.pt"

    def save(self, tag="state"):
        torch.save({"model": self.raw_model.state_dict(),
                    "opt": self.opt.state_dict(),
                    "sched": self.sched.state_dict(),
                    "scaler": self.scaler.state_dict(),
                    "state": self.state,
                    "torch_rng": torch.get_rng_state(),
                    "np_rng": np.random.get_state()},
                   self.out / (tag + ".pt"))
        (self.out / "state.json").write_text(json.dumps(self.state, indent=2, default=str))

    def load(self):
        if not self.ckpt.exists():
            return False
        try:
            d = torch.load(self.ckpt, map_location=self.device, weights_only=False)
            self.raw_model.load_state_dict(d["model"])
            self.opt.load_state_dict(d["opt"]); self.sched.load_state_dict(d["sched"])
            self.scaler.load_state_dict(d["scaler"]); self.state = d["state"]
            try:
                torch.set_rng_state(d["torch_rng"].cpu()); np.random.set_state(d["np_rng"])
            except Exception:
                pass
            print(f"  resumed {self.run_id} at epoch {self.state['epoch']}")
            return True
        except Exception as e:
            print(f"  checkpoint unreadable ({type(e).__name__}), starting fresh")
            return False

    def _loader(self, ds, shuffle):
        if len(ds) == 0:
            raise RuntimeError("empty dataset -- check the fold split; training on nothing "
                               "would produce a plausible-looking but untrained checkpoint")
        # drop_last=True on a split smaller than one batch yields ZERO batches, the optimiser
        # never steps, and the loss is reported as 0.00000. Guard it explicitly.
        drop = bool(shuffle) and len(ds) > self.bs
        if shuffle and not drop:
            print(f"  note: only {len(ds)} train window(s) < batch {self.bs}; keeping the "
                  f"partial batch so the optimiser actually steps")
        return DataLoader(ds, batch_size=min(self.bs, max(len(ds), 1)), shuffle=shuffle,
                          num_workers=self.nw, pin_memory=(self.device.type == "cuda"),
                          drop_last=drop, persistent_workers=self.nw > 0)

    def _step(self, batch, train):
        x, y, pk, rr = [b.to(self.device, non_blocking=True) for b in batch]
        with _autocast(self.device.type, self.amp):
            pred = self.model(x)
            if isinstance(pred, dict) and "aux" in pred and not train:
                pred = {k: v for k, v in pred.items() if k != "aux"}
            loss, parts = self.loss_fn(pred, y, pk, rr)
        return loss, parts, pred, y

    def fit(self, train_ds, val_ds):
        tl = self._loader(train_ds, True)
        vl = self._loader(val_ds, False)
        start = self.state["epoch"]
        # Only the epoch counter decides completion. Keying off a sticky `done` flag meant
        # raising CFG["EPOCHS"] later silently no-opped instead of training further.
        if start >= self.epochs:
            print(f"  {self.run_id} already complete at epoch {start}/{self.epochs}")
            return self.state
        if self.state.get("done"):
            print(f"  extending {self.run_id}: {start} -> {self.epochs} epochs")
            self.state["done"] = False
        bad = 0
        for ep in range(start, self.epochs):
            self.model.train(); t0 = time.time(); tot = 0.0; n = 0
            for i, batch in enumerate(tl):
                self.opt.zero_grad(set_to_none=True)
                loss, parts, _, _ = self._step(batch, True)
                self.scaler.scale(loss).backward()
                if self.grad_clip:
                    self.scaler.unscale_(self.opt)
                    torch.nn.utils.clip_grad_norm_(self.raw_model.parameters(), self.grad_clip)
                self.scaler.step(self.opt); self.scaler.update()
                tot += float(loss.detach()); n += 1
            if n == 0:
                raise RuntimeError(
                    "the training loader yielded zero batches -- the optimiser never stepped. "
                    "This would write a checkpoint that looks trained and is not.")
            self.sched.step()
            tr = tot / n
            self.model.eval(); vtot = 0.0; vn = 0
            with torch.no_grad():
                for batch in vl:
                    loss, _, _, _ = self._step(batch, False)
                    vtot += float(loss); vn += 1
            va = vtot / max(vn, 1)
            rec = {"epoch": ep + 1, "train": tr, "val": va,
                   "lr": self.opt.param_groups[0]["lr"], "sec": round(time.time() - t0, 1)}
            self.state["history"].append(rec); self.state["epoch"] = ep + 1
            improved = va < self.state["best"] - 1e-6
            if improved:
                self.state["best"] = va; self.state["best_epoch"] = ep + 1; bad = 0
                self.save("best")
            else:
                bad += 1
            self.save("state")
            if self.sync:
                self.sync.log("epoch", run=self.run_id, **rec, best=round(self.state["best"], 6))
                if improved:
                    self.sync.stage_done(f"{self.run_id}:best@{ep+1}")
            print(f"  ep {ep+1:>3}/{self.epochs}  train {tr:.5f}  val {va:.5f}"
                  f"{'  *' if improved else ''}  {rec['sec']:.0f}s")
            if bad >= self.patience:
                print(f"  early stop at epoch {ep+1} (no improvement for {self.patience})")
                break
        self.state["done"] = True; self.save("state")
        if self.sync:
            self.sync.stage_done(f"{self.run_id}:done")
        return self.state

    @torch.no_grad()
    def predict(self, ds, max_keep=200):
        bp = self.out / "best.pt"
        if bp.exists():
            try:
                self.raw_model.load_state_dict(
                    torch.load(bp, map_location=self.device, weights_only=False)["model"])
            except Exception as e:
                print("  could not load best.pt:", e)
        self.model.eval()
        dl = self._loader(ds, False)
        Y, P = [], []
        for batch in dl:
            x, y, pk, rr = [b.to(self.device, non_blocking=True) for b in batch]
            with _autocast(self.device.type, self.amp):
                out = self.model(x)
            Y.append(y.squeeze(1).float().cpu().numpy())
            P.append(out["wave"].squeeze(1).float().cpu().numpy())
        Y = np.concatenate(Y, 0); P = np.concatenate(P, 0)
        return Y, P
'''
