"""Signal and noise simulation for the spectrum-sensing study (vectorised, complex baseband).

Conventions
- All generators return complex arrays of shape (n, N) with unit average power per example
  (signals, surrogates) or unit expected power (noise), unless stated otherwise.
- SNR is defined as signal power / average noise power of the *white* noise added under H1.
- Randomness always comes from an explicit numpy Generator; callers derive independent streams
  for train / calibration / test with `streams(seed)`.
"""
import numpy as np
from scipy import signal as ss

MODS = ("BPSK", "QPSK", "8PSK", "QAM16", "QAM64", "PAM4")


def streams(seed):
    """Independent RNG streams for training, calibration and test data (no shared draws)."""
    ss_ = np.random.SeedSequence(seed)
    tr, cal, te = ss_.spawn(3)
    return np.random.default_rng(tr), np.random.default_rng(cal), np.random.default_rng(te)


# ----------------------------------------------------------------------------- pulses / symbols
_RRC_CACHE = {}


def rrc(beta=0.35, sps=8, span=8):
    key = (round(beta, 6), sps, span)
    if key in _RRC_CACHE:
        return _RRC_CACHE[key]
    t = np.arange(-span * sps, span * sps + 1) / sps
    h = np.empty_like(t)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-9:
            h[i] = 1 - beta + 4 * beta / np.pi
        elif abs(abs(ti) - 1 / (4 * beta)) < 1e-9:
            h[i] = beta / np.sqrt(2) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * beta)) + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
        else:
            h[i] = (np.sin(np.pi * ti * (1 - beta)) + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta))) / (np.pi * ti * (1 - (4 * beta * ti) ** 2))
    h = h / np.sqrt(np.sum(h ** 2))
    _RRC_CACHE[key] = h
    return h


def symbols(mod, shape, rng):
    if mod == "BPSK":
        s = rng.choice(np.array([-1.0, 1.0]), shape).astype(complex)
    elif mod == "QPSK":
        s = np.exp(1j * (np.pi / 4 + np.pi / 2 * rng.integers(0, 4, shape)))
    elif mod == "8PSK":
        s = np.exp(1j * np.pi / 4 * rng.integers(0, 8, shape))
    elif mod == "PAM4":
        s = rng.choice(np.array([-3.0, -1.0, 1.0, 3.0]), shape).astype(complex)
    elif mod in ("QAM16", "QAM64"):
        m = 4 if mod == "QAM16" else 8
        lv = np.arange(-(m - 1), m, 2).astype(float)
        s = rng.choice(lv, shape) + 1j * rng.choice(lv, shape)
    else:
        raise ValueError(mod)
    return s / np.sqrt(np.mean(np.abs(s) ** 2))


def modulated(n, N, rng, sps=8, beta=0.35, mods=MODS, cfo_max=0.01):
    """Linearly modulated RRC signals: random modulation per example, random symbol timing,
    random carrier phase and CFO ~ U[-cfo_max, cfo_max] (cycles/sample). Unit power per example."""
    h = rrc(beta, sps)
    span_sym = len(h) // sps + 2
    nsym = int(np.ceil(N / sps)) + 2 * span_sym + 2
    which = rng.integers(len(mods), size=n)
    sym = np.empty((n, nsym), complex)
    for k, mod in enumerate(mods):
        idx = np.flatnonzero(which == k)
        if idx.size:
            sym[idx] = symbols(mod, (idx.size, nsym), rng)
    up = np.zeros((n, nsym * sps), complex)
    up[:, ::sps] = sym
    x = ss.fftconvolve(up, h[None, :], mode="same", axes=1)
    off = span_sym * sps + rng.integers(0, sps, size=n)
    cols = off[:, None] + np.arange(N)[None, :]
    x = np.take_along_axis(x, cols, axis=1)
    k = np.arange(N)[None, :]
    x = x * np.exp(1j * (2 * np.pi * rng.uniform(-cfo_max, cfo_max, (n, 1)) * k + rng.uniform(0, 2 * np.pi, (n, 1))))
    return x / np.sqrt(np.mean(np.abs(x) ** 2, axis=1, keepdims=True))


# ----------------------------------------------------------------------------- noise families
def white(n, N, rng, power=1.0):
    return np.sqrt(power / 2) * (rng.standard_normal((n, N)) + 1j * rng.standard_normal((n, N)))


def from_psd(n, N, rng, psd):
    """Stationary-like complex Gaussian noise with a prescribed PSD shape on the N-point DFT grid
    (circulant synthesis). psd: array (N,) over np.fft.fftfreq(N) order, or callable f -> psd.
    Expected average power 1."""
    f = np.fft.fftfreq(N)
    p = psd(f) if callable(psd) else np.asarray(psd, float)
    p = p / p.mean()
    W = (rng.standard_normal((n, N)) + 1j * rng.standard_normal((n, N))) / np.sqrt(2)
    return np.fft.ifft(W * np.sqrt(p)[None, :], axis=1) * np.sqrt(N)


def ar1(n, N, rng, rho):
    """AR(1)-coloured complex Gaussian noise, unit power; rho scalar or per-example array.
    PSD tilt DC-to-band-edge = 20 log10((1+rho)/(1-rho)) dB (low-pass for rho>0)."""
    rho = np.broadcast_to(np.asarray(rho, float), (n,)).copy()
    w = white(n, N, rng)
    x = np.empty_like(w)
    x[:, 0] = w[:, 0] / np.sqrt(1 - rho ** 2)
    for k in range(1, N):
        x[:, k] = rho * x[:, k - 1] + w[:, k]
    return x * np.sqrt(1 - rho ** 2)[:, None]


def pink(n, N, rng):
    return from_psd(n, N, rng, lambda f: 1.0 / np.maximum(np.abs(f), 1.0 / N))


def butter_rolloff(n, N, rng, cutoff, order=6):
    """White noise through a digital Butterworth low-pass (receiver channel/anti-alias filter)."""
    b, a = ss.butter(order, cutoff)
    x = ss.lfilter(b, a, white(n, 2 * N, rng), axis=1)[:, N:]
    return x / np.sqrt(np.mean(np.abs(x) ** 2))


def rtl_like_psd(f, edge_db=3.0):
    """Smooth band-edge roll-off resembling the measured RTL-SDR noise PSD shape
    (0 dB at centre, about -edge_db at |f| = 0.5; cf. Mariani, Giorgetti, Chiani 2019, Fig. 1)."""
    return 10 ** (-edge_db * (np.abs(f) / 0.5) ** 4 / 10)


def rtl_like(n, N, rng, edge_db=3.0):
    return from_psd(n, N, rng, lambda f: rtl_like_psd(f, edge_db))


def band_tilt_psd(f, R_db, center=0.0, half_width=0.0844, sign=+1):
    """Two-level shape: level R^{+-1/2} inside the band |f-center| <= half_width, R^{-+1/2} outside
    (peak-to-peak ratio R). With sign=+1 the noise is raised in the band (signal-like)."""
    R = 10 ** (R_db / 10)
    inside = np.abs(((f - center + 0.5) % 1.0) - 0.5) <= half_width
    return np.where(inside, R ** (0.5 * sign), R ** (-0.5 * sign))


def band_tilt(n, N, rng, R_db, sign=+1, half_width=0.0844):
    return from_psd(n, N, rng, lambda f: band_tilt_psd(f, R_db, half_width=half_width, sign=sign))


def surrogate(n, N, rng, sps=8, beta=0.35, cfo_max=0.01):
    """PSD-matched Gaussian surrogate of `modulated`: stationary Gaussian process with the RRC
    signal's PSD and the same CFO law; no constellation or cyclostationary structure."""
    h = rrc(beta, sps)
    L = len(h)
    w = white(n, N + 2 * L, rng)
    x = ss.fftconvolve(w, h[None, :], mode="same", axes=1)[:, L: L + N]
    k = np.arange(N)[None, :]
    x = x * np.exp(1j * 2 * np.pi * rng.uniform(-cfo_max, cfo_max, (n, 1)) * k)
    return x / np.sqrt(np.mean(np.abs(x) ** 2, axis=1, keepdims=True))


def h1(sig, snr_db, rng):
    """Signal (unit power) at SNR snr_db (scalar or (n,)) in unit-power white noise."""
    snr = np.asarray(snr_db, float)
    g = np.sqrt(10 ** (snr / 10))
    g = g[:, None] if g.ndim == 1 else g
    return sig * g + white(sig.shape[0], sig.shape[1], rng)


# Test H0 shapes used throughout (name -> generator(n, N, rng))
TEST_SHAPES = {
    "white": lambda n, N, r: white(n, N, r),
    "white+1dB": lambda n, N, r: white(n, N, r, 10 ** 0.1),
    "white-1dB": lambda n, N, r: white(n, N, r, 10 ** -0.1),
    "ar+0.05": lambda n, N, r: ar1(n, N, r, 0.05),
    "ar-0.05": lambda n, N, r: ar1(n, N, r, -0.05),
    "ar+0.1": lambda n, N, r: ar1(n, N, r, 0.1),
    "ar-0.1": lambda n, N, r: ar1(n, N, r, -0.1),
    "ar+0.2": lambda n, N, r: ar1(n, N, r, 0.2),
    "ar-0.2": lambda n, N, r: ar1(n, N, r, -0.2),
    "pink": lambda n, N, r: pink(n, N, r),
    "lpf0.9": lambda n, N, r: butter_rolloff(n, N, r, 0.9),
    "lpf0.8": lambda n, N, r: butter_rolloff(n, N, r, 0.8),
    "rtl1dB": lambda n, N, r: rtl_like(n, N, r, 1.0),
    "rtl3dB": lambda n, N, r: rtl_like(n, N, r, 3.0),
}

# Declared uncertainty sets for robust (worst-case) calibration
UNCERTAINTY_SETS = {
    "U_power": ["white", "white+1dB", "white-1dB"],
    "U_mild": ["white", "ar+0.05", "ar-0.05", "ar+0.1", "ar-0.1", "lpf0.9", "rtl1dB"],
    "U_receiver": ["white", "lpf0.9", "lpf0.8", "rtl1dB", "rtl3dB"],
    "U_broad": ["white", "ar+0.1", "ar-0.1", "ar+0.2", "ar-0.2", "pink", "lpf0.9", "lpf0.8", "rtl3dB"],
}


def occupancy(sps, beta=0.35):
    return min(1.0, (1 + beta) / sps)


def shape_wall(rho, R_db):
    """Closed-form noise-shape SNR wall (linear) for scale-invariant tests of a flat-in-band
    Gaussian signal with occupancy rho under peak-to-peak noise-PSD uncertainty R_db."""
    R = 10 ** (R_db / 10)
    return rho * (R * R - 1) / (rho + R * (1 - rho))


def ed_power_wall(u_db):
    """Tandra-Sahai energy-detector wall (linear SNR w.r.t. the nominal = geometric-centre noise power)
    when the noise power is only known to lie in an interval of PEAK-TO-PEAK width u_db, i.e.
    sigma^2 in [sigma_n^2 / r, r sigma_n^2] with r = 10^(u_db/20). Wall = (r^2 - 1)/r = (R - 1)/sqrt(R),
    R = 10^(u_db/10). (Tandra & Sahai quote x = u_db/2 as the one-sided uncertainty.)"""
    R = 10 ** (u_db / 10)
    return (R - 1) / np.sqrt(R)


def ofdm(n, N, rng, n_sub=64, n_cp=16, occupied_sub=48, multipath=True, cfo_max=0.01):
    """Multi-carrier OFDM signals (e.g., 5G/Wi-Fi sub-band sensing):
    n_sub total subcarriers, n_cp cyclic prefix, occupied_sub active data subcarriers (occupancy rho = occupied_sub / n_sub).
    Random QPSK symbols per active subcarrier, IFFT, cyclic prefix insertion, Rayleigh multipath channel,
    random timing offset, phase and carrier frequency offset (CFO). Normalized to unit power per example."""
    T = n_sub + n_cp
    n_syms = int(np.ceil((N + T) / T)) + 3
    half_occ = occupied_sub // 2
    pos_idx = np.arange(1, half_occ + 1)
    neg_idx = np.arange(n_sub - half_occ, n_sub)
    active_idx = np.concatenate([pos_idx, neg_idx])

    S = np.zeros((n, n_syms, n_sub), dtype=complex)
    syms = np.exp(1j * (np.pi / 4 + np.pi / 2 * rng.integers(0, 4, (n, n_syms, len(active_idx)))))
    S[:, :, active_idx] = syms

    x_body = np.fft.ifft(S, axis=-1) * np.sqrt(n_sub)
    x_cp = x_body[:, :, -n_cp:]
    x_sym = np.concatenate([x_cp, x_body], axis=-1)
    x_stream = x_sym.reshape(n, -1)

    if multipath:
        pdp = np.exp(-np.arange(4) / 1.5)
        pdp = pdp / np.sqrt(np.sum(pdp ** 2))
        h_taps = (rng.standard_normal((n, 4)) + 1j * rng.standard_normal((n, 4))) / np.sqrt(2) * pdp[None, :]
        filtered = []
        for i in range(n):
            filtered.append(ss.fftconvolve(x_stream[i], h_taps[i], mode="same"))
        x_stream = np.stack(filtered)

    max_start = x_stream.shape[1] - N - 1
    starts = rng.integers(T, max_start, size=n)
    cols = starts[:, None] + np.arange(N)[None, :]
    x_win = np.take_along_axis(x_stream, cols, axis=1)

    cfo = rng.uniform(-cfo_max, cfo_max, (n, 1))
    t = np.arange(N)[None, :]
    x_win = x_win * np.exp(1j * (2 * np.pi * cfo * t + rng.uniform(0, 2 * np.pi, (n, 1))))

    pwr = np.sqrt(np.mean(np.abs(x_win) ** 2, axis=1, keepdims=True))
    return x_win / np.maximum(pwr, 1e-12)


def ofdm_surrogate(n, N, rng, n_sub=64, n_cp=16, occupied_sub=48, multipath=True, cfo_max=0.01):
    """Per-window Fourier phase-randomised surrogate of OFDM signals (docstring corrected 2026-10-05: it is NOT a context
    surrogate): keeps each window's periodogram exactly, but makes the window circular (periodic); destroys subcarrier
    phase alignment, CP correlation and OFDM frame structure."""
    sig = ofdm(n, N, rng, n_sub=n_sub, n_cp=n_cp, occupied_sub=occupied_sub, multipath=multipath, cfo_max=cfo_max)
    S = np.fft.fft(sig, axis=1)
    sur = np.fft.ifft(S * np.exp(2j * np.pi * rng.random(S.shape)), axis=1)
    return sur / np.sqrt(np.mean(np.abs(sur) ** 2, axis=1, keepdims=True))

