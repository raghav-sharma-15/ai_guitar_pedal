# scripts/preprocess_audio.py

"""
Module: preprocess_audio
Description:
    Extract either hand-crafted features or log-mel spectrograms for the GTZAN dataset.
"""

import os
import numpy as np
import librosa
import soundfile as sf
import argparse

# GTZAN genres
GENRES = [
    'blues','classical','country','disco','hiphop',
    'jazz','metal','pop','reggae','rock'
]


def extract_features_from_array(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Hand-crafted features (84 dims): MFCC, Δ/ΔΔ MFCC, Chroma,
    Contrast, Bandwidth, ZCR, Centroid, Rolloff, Tempo.
    """
    if y.ndim > 1:
        y = np.mean(y, axis=1)

    # MFCC + deltas
    mfcc       = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2= librosa.feature.delta(mfcc, order=2)
    # Other features
    chroma     = librosa.feature.chroma_stft(y=y, sr=sr)
    contrast   = librosa.feature.spectral_contrast(y=y, sr=sr)
    bandwidth  = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    zcr        = librosa.feature.zero_crossing_rate(y)
    centroid   = librosa.feature.spectral_centroid(y=y, sr=sr)
    rolloff    = librosa.feature.spectral_rolloff(y=y, sr=sr)
    tempo_est, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(tempo_est)

    feats = []
    feats += np.mean(mfcc,       axis=1).tolist()
    feats += np.mean(mfcc_delta, axis=1).tolist()
    feats += np.mean(mfcc_delta2,axis=1).tolist()
    feats += np.mean(chroma,     axis=1).tolist()
    feats += np.mean(contrast,   axis=1).tolist()
    feats.append(float(np.mean(bandwidth)))
    feats.append(float(np.mean(zcr)))
    feats.append(float(np.mean(centroid)))
    feats.append(float(np.mean(rolloff)))
    feats.append(tempo)
    return np.array(feats, dtype=np.float32)


def extract_spectrogram(y: np.ndarray, sr: int,
                        n_mels: int=64, hop_length: int=512) -> np.ndarray:
    """
    Compute log-mel spectrogram for CNN input.
    Returns shape (n_mels, T).
    """
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
                                         hop_length=hop_length)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel.astype(np.float32)


def process_features(input_dir: str, output_file: str):
    features, labels = [], []
    for genre in GENRES:
        dirg = os.path.join(input_dir, genre)
        if not os.path.isdir(dirg): continue
        for f in sorted(os.listdir(dirg)):
            if not f.endswith('.wav'): continue
            path = os.path.join(dirg, f)
            try:
                y, _ = librosa.load(path, sr=22050, mono=True)
                feat = extract_features_from_array(y, 22050)
                features.append(feat)
                labels.append(GENRES.index(genre))
            except Exception:
                print(f"⚠️ Skipping {path}")
    np.save(output_file, {'features':np.stack(features),
                          'labels':np.array(labels, dtype=np.int64)})
    print(f"✅ Saved features: {len(features)} samples to {output_file}")


def process_spectrograms(input_dir: str, output_file: str,
                         duration: float=5.0,
                         n_mels: int=64, hop_length: int=512):
    specs, labels = [], []
    for genre in GENRES:
        dirg = os.path.join(input_dir, genre)
        if not os.path.isdir(dirg): continue
        for f in sorted(os.listdir(dirg)):
            if not f.endswith('.wav'): continue
            path = os.path.join(dirg, f)
            try:
                y, _ = librosa.load(path, sr=22050, mono=True,
                                     duration=duration)
                spec = extract_spectrogram(y, 22050, n_mels, hop_length)
                specs.append(spec)
                labels.append(GENRES.index(genre))
            except Exception:
                print(f"⚠️ Skipping {path}")
    specs = np.stack(specs)[:, None, :, :]
    np.savez(output_file, specs=specs,
             labels=np.array(labels, dtype=np.int64))
    print(f"✅ Saved spectrograms: {len(labels)} samples to {output_file}")


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description="Extract features or spectrograms from GTZAN"
    )
    p.add_argument('mode', choices=['features','spectrogram'],
                   help='Extraction mode')
    p.add_argument('input_dir', nargs='?', default='data/gtzan',
                   help='GTZAN root folder')
    p.add_argument('--output','-o', default=None,
                   help='Output file (defaults based on mode)')
    args = p.parse_args()
    if args.mode == 'features':
        out = args.output or 'data/features.npy'
        process_features(args.input_dir, out)
    else:
        out = args.output or 'data/specs.npz'
        process_spectrograms(args.input_dir, out)
