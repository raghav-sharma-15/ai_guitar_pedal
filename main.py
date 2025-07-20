# main.py

"""
Offline AI Guitar Pedal:
  • Load WAV, extract & scale features
  • Predict genre via trained model
  • Apply genre-specific effects
  • Save processed WAV
"""

import argparse
import os
import torch
import joblib
import soundfile as sf
from scripts.preprocess_audio import extract_features
from scripts.effects import apply_genre_effects
from scripts.train_model import GenreNet

# GTZAN genres (must match preprocess & effects)
GENRES = [
    'blues', 'classical', 'country', 'disco', 'hiphop',
    'jazz', 'metal', 'pop', 'reggae', 'rock'
]

def load_model(path, input_dim, num_classes, device):
    model = GenreNet(input_dim, num_classes).to(device)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model

def main(args):
    # Extract features
    feat = extract_features(args.input)
    if feat is None:
        print(f"❌ Could not load '{args.input}'")
        return

    # Load scaler & scale
    scaler = joblib.load(args.scaler)
    feat_scaled = scaler.transform(feat.reshape(1, -1))

    # Load model & predict
    device = args.device
    feats = torch.tensor(feat_scaled, dtype=torch.float32).to(device)
    model = load_model(args.model, feats.shape[1], len(GENRES), device)
    with torch.no_grad():
        pred = torch.argmax(model(feats), dim=1).item()
    genre = GENRES[pred]
    print(f"Detected genre: {genre}")

    # Read, process, and write audio
    y, sr = sf.read(args.input)
    y_fx = apply_genre_effects(y, sr, genre)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    sf.write(args.output, y_fx, sr)
    print(f"✅ Processed file saved to {args.output}")

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Offline AI Guitar Pedal')
    p.add_argument('--input',  required=True, help='Input WAV path')
    p.add_argument('--output', required=True, help='Output WAV path')
    p.add_argument('--model',  default='models/genre_model_best.pth', help='Model checkpoint')
    p.add_argument('--scaler', default='models/scaler.pkl',       help='Feature scaler path')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='Torch device')
    main(p.parse_args())