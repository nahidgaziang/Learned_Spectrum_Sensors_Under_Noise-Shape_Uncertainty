"""Classical and learned spectrum-sensing detectors.

Every detector maps complex inputs x (n, N) to a real score (n,): larger = more signal-like.
Thresholds are set elsewhere (calibration), never inside a detector.
"""
import numpy as np
import torch
import torch.nn as nn


# ============================================================================ classical
def ed(x, noise_power=1.0):
    """Energy detector with an assumed noise power (not scale-invariant)."""
    return np.mean(np.abs(x) ** 2, axis=1) / noise_power


def _acov(x, L):
    """Sample autocorrelation r[k] = mean_n x[n+k] conj(x[n]), k = 0..L-1, via FFT (n, L)."""
    n, N = x.shape
    nfft = 1 << int(np.ceil(np.log2(2 * N)))
    X = np.fft.fft(x, nfft, axis=1)
    r = np.fft.ifft(np.abs(X) ** 2, axis=1)[:, :L]
    return r / (N - np.arange(L))[None, :]


def mme(x, L=16):
    """Max/min eigenvalue ratio of the L x L Toeplitz sample covariance built from the sample
    autocorrelation (scale-invariant; memory-safe for long N)."""
    r = _acov(x, L)
    lag = np.arange(L)[:, None] - np.arange(L)[None, :]
    R = np.where(lag[None] >= 0, r[:, np.abs(lag)], np.conj(r[:, np.abs(lag)]))
    ev = np.linalg.eigvalsh(R)
    return ev[:, -1] / np.maximum(ev[:, 0], 1e-12)


def sfm(x, nseg=8):
    """Negative spectral flatness (geometric/arithmetic mean of a Bartlett periodogram);
    scale-invariant second-order statistic. Larger = less flat = more signal-like."""
    n, N = x.shape
    m = N // nseg
    seg = x[:, : m * nseg].reshape(n, nseg, m)
    P = np.mean(np.abs(np.fft.fft(seg, axis=2)) ** 2, axis=1) + 1e-12
    return -np.exp(np.mean(np.log(P), axis=1)) / np.mean(P, axis=1)


def cyc(x, sps=8, lags=(0, 1, 2, 3, 4)):
    """Normalised cyclic-autocorrelation detector at cycle frequency alpha = 1/sps (symbol rate).
    Stationary noise of any shape has zero cyclic autocorrelation at alpha != 0 (asymptotically)."""
    n, N = x.shape
    alpha = 1.0 / sps
    ph = np.exp(-1j * 2 * np.pi * alpha * np.arange(N))
    p0 = np.mean(np.abs(x) ** 2, axis=1)
    s = np.zeros(n)
    for tau in lags:
        prod = x[:, tau:] * np.conj(x[:, : N - tau]) * ph[None, : N - tau]
        s += np.abs(np.mean(prod, axis=1)) ** 2
    return s / p0 ** 2


def band_ratio(x, half_width=0.0844):
    """Oracle second-order test with known signal band: periodogram power inside |f|<=hw over outside."""
    n, N = x.shape
    f = np.fft.fftfreq(N)
    P = np.abs(np.fft.fft(x, axis=1)) ** 2
    inb = np.abs(f) <= half_width
    return P[:, inb].mean(1) / np.maximum(P[:, ~inb].mean(1), 1e-12)


CLASSICAL = {"ED": ed, "MME": mme, "SFM": sfm, "CYC": cyc, "BAND": band_ratio}


# ============================================================================ preprocessing
def norm_iq(x):
    """Per-example power normalisation (the scale-invariance shared by the DL recipes)."""
    x = x / np.sqrt(np.mean(np.abs(x) ** 2, axis=1, keepdims=True))
    return np.stack([x.real, x.imag], axis=1).astype(np.float32)


def cov_input(x, L=16):
    """Normalised L x L smoothed sample covariance as a 2-channel image (real, imag)."""
    x = x / np.sqrt(np.mean(np.abs(x) ** 2, axis=1, keepdims=True))
    n, N = x.shape
    M = N - L + 1
    idx = np.arange(L)[:, None] + np.arange(M)[None, :]
    Y = x[:, idx]
    R = np.einsum("nim,njm->nij", Y, Y.conj()) / M
    return np.stack([R.real, R.imag], axis=1).astype(np.float32)


def stft_input(x, nfft=16, hop=8):
    """Log-magnitude STFT (Hann) of the normalised signal, shape (n, 1, nfft, frames)."""
    x = x / np.sqrt(np.mean(np.abs(x) ** 2, axis=1, keepdims=True))
    n, N = x.shape
    win = np.hanning(nfft)
    starts = np.arange(0, N - nfft + 1, hop)
    frames = np.stack([x[:, s:s + nfft] * win for s in starts], axis=2)   # (n, nfft, T)
    S = np.abs(np.fft.fftshift(np.fft.fft(frames, axis=1), axes=1)) ** 2
    return np.log(S + 1e-6)[:, None].astype(np.float32)


def cyc_input(x, sps=8, num_lags=16):
    """Normalised cyclic autocorrelation vector at alpha = 1/sps (real and imag parts), shape (n, 2, num_lags)."""
    n, N = x.shape
    alpha = 1.0 / sps
    ph = np.exp(-1j * 2 * np.pi * alpha * np.arange(N))
    p0 = np.mean(np.abs(x) ** 2, axis=1) + 1e-12
    lags = np.empty((n, num_lags), dtype=np.complex128)
    for tau in range(num_lags):
        prod = x[:, tau:] * np.conj(x[:, : N - tau]) * ph[None, : N - tau]
        lags[:, tau] = np.mean(prod, axis=1) / p0
    return np.stack([lags.real, lags.imag], axis=1).astype(np.float32)


# ============================================================================ learned models
class CLDNN(nn.Module):
    """DetectNet-like CNN + LSTM (Gao et al. 2019 recipe)."""
    rep = "iq"

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(2, 60, 10, padding=5), nn.ReLU(), nn.Dropout(0.2),
                                  nn.Conv1d(60, 60, 10, padding=4), nn.ReLU(), nn.Dropout(0.2))
        self.lstm = nn.LSTM(60, 128, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 1))

    def forward(self, x):
        h = self.conv(x).transpose(1, 2)
        _, (hn, _) = self.lstm(h)
        return self.fc(hn[-1]).squeeze(-1)


class _Res(nn.Module):
    def __init__(self, c_in, c_out, stride):
        super().__init__()
        self.b = nn.Sequential(nn.Conv1d(c_in, c_out, 3, stride, 1), nn.BatchNorm1d(c_out), nn.ReLU(),
                               nn.Conv1d(c_out, c_out, 3, 1, 1), nn.BatchNorm1d(c_out))
        self.s = (nn.Sequential(nn.Conv1d(c_in, c_out, 1, stride), nn.BatchNorm1d(c_out))
                  if (stride != 1 or c_in != c_out) else nn.Identity())

    def forward(self, x):
        return torch.relu(self.b(x) + self.s(x))


class ResNet1D(nn.Module):
    """Residual 1-D CNN in the spirit of Zheng et al. 2019 (Table I), scaled to N."""
    rep = "iq"

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(2, 48, 15, padding=7), nn.ReLU(), nn.Conv1d(48, 64, 7, padding=3), nn.ReLU(),
                                  nn.MaxPool1d(2))
        self.body = nn.Sequential(_Res(64, 64, 2), _Res(64, 64, 1), _Res(64, 128, 2), _Res(128, 128, 1))
        self.head = nn.Sequential(nn.Linear(128, 32), nn.ReLU(), nn.Dropout(0.5), nn.Linear(32, 1))

    def forward(self, x):
        return self.head(self.body(self.stem(x)).mean(-1)).squeeze(-1)


class CovCNN(nn.Module):
    """2-D CNN on the normalised sample covariance (CM-CNN-style input; Liu et al. 2019)."""
    rep = "cov"

    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(2, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
                               nn.MaxPool2d(2), nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        self.h = nn.Sequential(nn.Flatten(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.h(self.f(x)).squeeze(-1)


class STFTCNN(nn.Module):
    """2-D CNN on the log-magnitude STFT (Chen et al. 2021 style)."""
    rep = "stft"

    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
                               nn.MaxPool2d(2), nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        self.h = nn.Sequential(nn.Flatten(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.h(self.f(x)).squeeze(-1)


class GapCNN(nn.Module):
    """Size-agnostic 1-D CNN with global mean/std pooling (any N)."""
    rep = "iq"

    def __init__(self, ch=64):
        super().__init__()
        self.f = nn.Sequential(nn.Conv1d(2, ch, 9, padding=4), nn.ReLU(), nn.Conv1d(ch, ch, 9, padding=4), nn.ReLU(),
                               nn.Conv1d(ch, ch, 9, padding=4, stride=2), nn.ReLU(), nn.Conv1d(ch, ch, 9, padding=4), nn.ReLU())
        h_dim = 64 if ch <= 64 else ch
        self.h = nn.Sequential(nn.Linear(2 * ch, h_dim), nn.ReLU(), nn.Linear(h_dim, 1))

    def forward(self, x):
        z = self.f(x)
        return self.h(torch.cat([z.mean(-1), z.std(-1)], 1)).squeeze(-1)


class GapCNN4X(GapCNN):
    """4x channel width GapCNN (~1.84M parameters) for capacity scaling ablation."""
    rep = "iq"

    def __init__(self):
        super().__init__(ch=256)


class DetectNetExact(nn.Module):
    """Layer-by-layer PyTorch replica of DetectNet as released by Gao et al. (utils.DetectNet, Keras):
    Conv1D(60,10,'same',relu)+Dropout(.2) x2 (channels_first on (2,N)) -> Flatten -> Dense(N, linear) ->
    reshape (1,N) -> concat with the input rows (3,N) -> LSTM(128, return_sequences) -> LSTM(128) ->
    Dense(128, relu) -> output. Keras treats the (3, N) tensor as 3 time steps of N features. The 2-way
    softmax is replaced by an equivalent single logit; recurrent dropout (not available in cuDNN LSTM) is
    approximated by dropout between the two LSTM layers."""
    rep = "iq"

    def __init__(self, N=128):
        super().__init__()
        self.N = N
        self.conv = nn.Sequential(nn.Conv1d(2, 60, 10, padding="same"), nn.ReLU(), nn.Dropout(0.2),
                                  nn.Conv1d(60, 60, 10, padding="same"), nn.ReLU(), nn.Dropout(0.2))
        self.proj = nn.Linear(60 * N, N)
        self.lstm1 = nn.LSTM(N, 128, batch_first=True)
        self.drop = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(128, 128, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, x):
        z = self.proj(self.conv(x).flatten(1)).unsqueeze(1)          # (B, 1, N)
        seq = torch.cat([z, x], dim=1)                                # (B, 3, N): 3 steps of N features
        h, _ = self.lstm1(seq)
        _, (hn, _) = self.lstm2(self.drop(h))
        return self.fc(hn[-1]).squeeze(-1)


class CycNet(nn.Module):
    """Cyclic-feature network: maps normalised cyclic autocorrelation lags (2, num_lags) to detection logit.
    Asymptotically immune to stationary noise of arbitrary PSD shape."""
    rep = "cyc"

    def __init__(self, num_lags=16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(2, 32, 3, padding=1), nn.ReLU(),
            nn.Conv1d(32, 32, 3, padding=1), nn.ReLU(),
            nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.fc(self.conv(x)).squeeze(-1)


class Transformer(nn.Module):
    """Attention-based sensor (added 2026-10-05, the project plan U8): the power-normalised I/Q window is cut
    into non-overlapping patches of 8 samples (16 tokens at N = 128), each linearly embedded (d = 64) with a learned
    positional embedding, followed by 4 pre-norm Transformer encoder layers (4 heads, feed-forward 128, dropout 0.1),
    mean pooling over tokens and a linear logit. Positional embeddings are interpolated for other N."""
    rep = "iq"

    def __init__(self, patch=8, d=64, layers=4, heads=4, ff=128, n_tokens=16):
        super().__init__()
        self.patch = patch
        self.embed = nn.Linear(2 * patch, d)
        self.pos = nn.Parameter(torch.zeros(1, n_tokens, d))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d, heads, ff, dropout=0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1)

    def forward(self, x):                                  # x: (B, 2, N)
        B, _, N = x.shape
        t = x.reshape(B, 2, N // self.patch, self.patch).permute(0, 2, 1, 3).reshape(B, N // self.patch, 2 * self.patch)
        z = self.embed(t)
        pos = self.pos if z.shape[1] == self.pos.shape[1] else nn.functional.interpolate(
            self.pos.transpose(1, 2), size=z.shape[1], mode="linear", align_corners=False).transpose(1, 2)
        z = self.enc(z + pos)
        return self.head(self.norm(z).mean(1)).squeeze(-1)


ARCHS = {"CLDNN": CLDNN, "ResNet1D": ResNet1D, "CovCNN": CovCNN, "STFTCNN": STFTCNN, "GapCNN": GapCNN,
         "GapCNN4X": GapCNN4X, "DetectNetExact": DetectNetExact, "CycNet": CycNet, "Transformer": Transformer}
REPS = {"iq": norm_iq, "cov": cov_input, "stft": stft_input, "cyc": cyc_input}


def raw_iq(x):
    """Ablation: I/Q WITHOUT per-example power normalisation (noise power is 1 in simulation, so the
    network can see absolute energy)."""
    return np.stack([x.real, x.imag], axis=1).astype(np.float32)


def represent(arch_name, x, norm=True, sps=8):
    if not norm:
        if ARCHS[arch_name].rep != "iq":
            raise ValueError("no-normalisation ablation implemented for I/Q models only")
        return raw_iq(x)
    if ARCHS[arch_name].rep == "cyc":
        return REPS["cyc"](x, sps=sps)
    return REPS[ARCHS[arch_name].rep](x)


def n_params(model):
    return sum(p.numel() for p in model.parameters())
