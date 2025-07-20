# scripts/effects.py

"""
Module: effects
Description:
    Implements a suite of audio effects—reverb, distortion, chorus, wah, delay, compressor—
    and applies genre-specific chains.
"""

import numpy as np
from scipy.signal import fftconvolve

def ensure_mono(y: np.ndarray) -> np.ndarray:
    if y.ndim > 1:
        return np.mean(y, axis=1)
    return y

def normalize(y: np.ndarray) -> np.ndarray:
    return y / (np.max(np.abs(y)) + 1e-6)

def add_reverb(y: np.ndarray, sr: int, decay=0.5, wet=0.3) -> np.ndarray:
    ir_len = int(sr * 0.5)
    # Exponential-decay impulse response
    impulse = decay ** np.linspace(0, 1, ir_len)
    reverb  = fftconvolve(y, impulse, mode='full')[:len(y)]
    return normalize((1-wet)*y + wet*reverb)

def add_distortion(y: np.ndarray, gain=20, level=0.8) -> np.ndarray:
    distorted = np.tanh(gain * y)
    return normalize((1-level)*y + level*distorted)

def add_chorus(y: np.ndarray, sr: int, depth=0.002, rate=1.5) -> np.ndarray:
    y = ensure_mono(y)
    max_delay = int(depth * sr)
    lfo = (1 + np.sin(2*np.pi*rate*np.arange(len(y))/sr)) / 2
    delay_idxs = (lfo * max_delay).astype(int)
    out = np.zeros_like(y)
    for i in range(len(y)):
        d = delay_idxs[i]
        out[i] = y[i] + (y[i-d] if i-d >= 0 else 0)
    return normalize(out)

def add_wah(y: np.ndarray, sr: int, freq=2.0) -> np.ndarray:
    # Simple amplitude modulation as wah
    lfo = np.sin(2*np.pi*freq * np.arange(len(y))/sr)
    return normalize(y * (lfo + 1)/2)

def add_delay(y: np.ndarray, sr: int, time=0.3, feedback=0.5, mix=0.5) -> np.ndarray:
    d = int(time * sr)
    buf = np.zeros(len(y) + d)
    buf[:len(y)] = y
    for i in range(len(y)):
        buf[i+d] += y[i] * feedback
    delayed = buf[:len(y)]
    return normalize((1-mix)*y + mix*delayed)

def add_compressor(y: np.ndarray, threshold=0.1, ratio=4) -> np.ndarray:
    abs_y = np.abs(y)
    over   = abs_y > threshold
    comp   = np.copy(y)
    comp[over] = np.sign(y[over]) * (threshold + (abs_y[over]-threshold)/ratio)
    return normalize(comp)

def apply_genre_effects(y: np.ndarray, sr: int, genre: str) -> np.ndarray:
    """
    Chains effects per genre:
      • jazz → chorus + reverb
      • rock → distortion + delay
      • metal → heavy distortion + compression
      • blues → overdrive + reverb
      • funk → wah + compression
      • clean → subtle reverb
    """
    y = ensure_mono(y)
    if genre == 'jazz':
        y = add_chorus(y, sr, depth=0.003, rate=1.2)
        y = add_reverb(y, sr, decay=0.4, wet=0.3)
    elif genre == 'rock':
        y = add_distortion(y, gain=30, level=0.7)
        y = add_delay(y, sr, time=0.25, feedback=0.3, mix=0.4)
    elif genre == 'metal':
        y = add_distortion(y, gain=40, level=1.0)
        y = add_compressor(y, threshold=0.2, ratio=8)
    elif genre == 'blues':
        y = add_distortion(y, gain=15, level=0.5)
        y = add_reverb(y, sr, decay=0.3, wet=0.4)
    elif genre == 'funk':
        y = add_wah(y, sr, freq=2.5)
        y = add_compressor(y, threshold=0.05, ratio=6)
    elif genre == 'clean':
        y = add_reverb(y, sr, decay=0.2, wet=0.2)
    return normalize(y)
