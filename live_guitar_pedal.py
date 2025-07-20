# live_guitar_pedal.py

"""
Real-time AI Guitar Pedal:
  • Duplex stream
  • Sliding window, genre classification + dynamic effects
  • Uses trained scaler & model for live predictions
"""

import numpy as np
import sounddevice as sd
import torch
import joblib
from collections import deque
from scripts.preprocess_audio import extract_features_from_array
from scripts.effects import apply_genre_effects
from scripts.train_model import GenreNet

# Constants
SR          = 22050
WINDOW_SEC  = 3
WINDOW_SAMP = SR * WINDOW_SEC
BLOCKSIZE   = 1024
GENRES      = [
    'blues', 'classical', 'country', 'disco', 'hiphop',
    'jazz', 'metal', 'pop', 'reggae', 'rock'
]

# Prepare model & scaler
device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_dummy    = np.zeros(WINDOW_SAMP, dtype=np.float32)
INPUT_DIM = len(extract_features_from_array(_dummy, SR))
model     = GenreNet(INPUT_DIM, len(GENRES)).to(device)
model.load_state_dict(torch.load('models/genre_model_best.pth', map_location=device))
model.eval()
scaler = joblib.load('models/scaler.pkl')

buf = deque(maxlen=WINDOW_SAMP)

def callback(indata, outdata, frames, time, status):
    if status:
        print(status)
    y = indata[:, 0]
    buf.extend(y)
    if len(buf) == WINDOW_SAMP:
        window = np.array(buf)
        feats = extract_features_from_array(window, SR)
        feats_scaled = scaler.transform(feats.reshape(1, -1))
        t = torch.tensor(feats_scaled, dtype=torch.float32).to(device)
        with torch.no_grad():
            genre_idx = torch.argmax(model(t), dim=1).item()
        genre = GENRES[genre_idx]
        y_fx = apply_genre_effects(y, SR, genre)
    else:
        y_fx = y

    outdata[:, 0] = y_fx

def main():
    print("🎸 Live AI Guitar Pedal — Press Ctrl+C to stop")
    with sd.Stream(channels=1, samplerate=SR, blocksize=BLOCKSIZE, callback=callback):
        sd.sleep(int(1e9))

if __name__ == '__main__':
    main()