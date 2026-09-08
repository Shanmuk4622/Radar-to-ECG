
# crvs_metrics.py -- every metric the baseline reports, plus the ones it should have.
import numpy as np
from scipy import signal as ss
from scipy import stats as sstats

def _f(x):
    return np.nan_to_num(np.asarray(x, np.float64), nan=0.0, posinf=0.0, neginf=0.0)

def pearson(a, b):
    a, b = _f(a), _f(b)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])

def psd(x, fs=128, nperseg=256):
    f, p = ss.welch(_f(x), fs=fs, nperseg=min(nperseg, len(x)))
    return f, p

def seg_metrics(y, yhat, fs=128):
    # One window. Correlations are reported x100 to match the baseline's tables.
    y, yhat = _f(y), _f(yhat)
    mae = float(np.mean(np.abs(y - yhat)))
    mse = float(np.mean((y - yhat) ** 2))
    cct = 100.0 * pearson(y, yhat)
    _, py = psd(y, fs); _, ph = psd(yhat, fs)
    ccs = 100.0 * pearson(py, ph)
    rms = lambda v: float(np.sqrt(np.mean(np.asarray(v, np.float64) ** 2)))
    rr_t = rms(yhat - y) / (rms(y) + 1e-12)
    rr_s = rms(ph - py) / (rms(py) + 1e-12)
    return {"MAE": mae, "MSE": mse, "CC_temporal": cct, "CC_spectral": ccs,
            "RRMSE_temporal": rr_t, "RRMSE_spectral": rr_s,
            "R2": float(1.0 - np.sum((y - yhat) ** 2) / (np.sum((y - y.mean()) ** 2) + 1e-12))}

def detect_r_peaks(x, fs=128, refractory_s=0.25):
    x = _f(x)
    if len(x) < int(2 * fs):
        return np.array([], int)
    ny = fs / 2.0
    sos = ss.butter(4, [5.0 / ny, min(25.0, ny * 0.95) / ny], btype="band", output="sos")
    b = ss.sosfiltfilt(sos, x)
    e = np.convolve(np.diff(b, prepend=b[0]) ** 2,
                    np.ones(max(1, int(0.10 * fs))) / max(1, int(0.10 * fs)), "same")
    thr = np.percentile(e, 98) * 0.35
    pk, _ = ss.find_peaks(e, height=thr, distance=max(1, int(refractory_s * fs)))
    return pk

def hrv_from_peaks(pk, fs=128):
    out = {"n_peaks": int(len(pk)), "mean_rr_ms": np.nan, "sd_rr_ms": np.nan,
           "mean_hr_bpm": np.nan, "sd_hr_bpm": np.nan, "rmssd_ms": np.nan}
    if len(pk) < 4:
        return out
    rr = np.diff(np.asarray(pk, float)) / fs * 1000.0
    rr = rr[(rr > 300) & (rr < 2000)]
    if len(rr) < 3:
        return out
    hr = 60000.0 / rr
    out.update(mean_rr_ms=float(rr.mean()), sd_rr_ms=float(rr.std()),
               mean_hr_bpm=float(hr.mean()), sd_hr_bpm=float(hr.std()),
               rmssd_ms=float(np.sqrt(np.mean(np.diff(rr) ** 2))))
    return out

def peak_detection_scores(y, yhat, fs=128, tol_ms=100.0):
    # Match predicted R peaks to ground-truth peaks within a tolerance window.
    gt = detect_r_peaks(y, fs); pr = detect_r_peaks(yhat, fs)
    tol = tol_ms / 1000.0 * fs
    used = np.zeros(len(pr), bool)
    tp = 0
    errs = []
    for g in gt:
        if len(pr) == 0:
            break
        # float, not the int64 that find_peaks returns -- assigning np.inf into an
        # integer array raises OverflowError even when the mask selects nothing.
        d = np.abs(pr - g).astype(np.float64)
        d[used] = np.inf
        j = int(np.argmin(d))
        if d[j] <= tol:
            tp += 1; used[j] = True; errs.append((pr[j] - g) / fs * 1000.0)
    fp = int((~used).sum()); fn = int(len(gt) - tp)
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return {"TP": tp, "FP": fp, "FN": fn, "precision": prec, "recall": rec, "F1": f1,
            "accuracy": tp / max(tp + fp + fn, 1),
            "timing_err_ms_median": float(np.median(np.abs(errs))) if errs else np.nan,
            "timing_err_ms_iqr": float(np.subtract(*np.percentile(np.abs(errs), [75, 25])))
                                  if len(errs) > 3 else np.nan,
            "missed_rate": fn / max(len(gt), 1)}

def aggregate(rows):
    import pandas as pd
    df = pd.DataFrame(rows)
    out = {}
    for c in df.columns:
        if df[c].dtype.kind in "fi":
            out[c] = float(df[c].mean()); out[c + "_std"] = float(df[c].std())
    return out

def bland_altman(a, b):
    a, b = _f(a), _f(b)
    m = (a + b) / 2.0; d = a - b
    bias = float(d.mean()); sd = float(d.std())
    return {"mean": m, "diff": d, "bias": bias, "sd": sd,
            "loa_lo": bias - 1.96 * sd, "loa_hi": bias + 1.96 * sd}

def wilcoxon_holm(groups, better="higher"):
    # Pairwise Wilcoxon signed-rank across folds, Holm-corrected. groups: {name: [values]}
    import itertools
    names = list(groups)
    raw = []
    for a, b in itertools.combinations(names, 2):
        x, y = np.asarray(groups[a], float), np.asarray(groups[b], float)
        n = min(len(x), len(y))
        if n < 3 or np.allclose(x[:n], y[:n]):
            raw.append((a, b, np.nan)); continue
        try:
            p = float(sstats.wilcoxon(x[:n], y[:n]).pvalue)
        except Exception:
            p = np.nan
        raw.append((a, b, p))
    ps = [r[2] for r in raw]
    order = np.argsort([p if np.isfinite(p) else 1.0 for p in ps])
    m = len(ps); adj = [np.nan] * m; run = 0.0
    for k, i in enumerate(order):
        p = ps[i]
        if not np.isfinite(p):
            continue
        run = max(run, (m - k) * p)
        adj[i] = min(1.0, run)
    return [{"a": raw[i][0], "b": raw[i][1], "p": ps[i], "p_holm": adj[i]} for i in range(m)]
