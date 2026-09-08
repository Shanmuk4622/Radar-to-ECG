
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
