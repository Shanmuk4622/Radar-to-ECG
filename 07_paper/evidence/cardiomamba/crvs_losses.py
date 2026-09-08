
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
