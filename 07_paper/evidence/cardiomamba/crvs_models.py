
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
