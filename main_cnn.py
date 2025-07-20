# main_cnn.py

import argparse
import torch
import numpy as np
import librosa
import soundfile as sf
from scripts.train_model import ConvNet   # Uses the ConvNet class in train_model.py
from scripts.effects import apply_genre_effects

GENRES = ['blues','classical','country','disco','hiphop',
          'jazz','metal','pop','reggae','rock']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def infer(input_path, model_path):
    y, _ = librosa.load(input_path, sr=22050, mono=True, duration=5.0)
    mel = librosa.feature.melspectrogram(y=y, sr=22050, n_mels=64, hop_length=512)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    spec = torch.tensor(log_mel[None,None,:,:], dtype=torch.float32).to(DEVICE)

    model = ConvNet(len(GENRES)).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    with torch.no_grad():
        pred = torch.argmax(model(spec), dim=1).item()
    return GENRES[pred]

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input',  required=True)
    p.add_argument('--model',  default='models/cnn_best.pth')
    p.add_argument('--output', required=True)
    args = p.parse_args()

    genre = infer(args.input, args.model)
    print(f"Detected genre: {genre}")

    y, sr = sf.read(args.input)
    y_fx = apply_genre_effects(y, sr, genre)
    sf.write(args.output, y_fx, sr)
    print(f"✅ Saved processed to {args.output}")