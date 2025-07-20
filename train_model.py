# scripts/train_model.py

"""
Module: train_model
Description:
    Train a genre classifier on either hand-crafted features (MLP) or
    log-mel spectrograms (CNN) for the GTZAN dataset.
"""
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib

# CONFIGURATION
BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-3
TEST_SIZE = 0.2
RANDOM_STATE = 42
MODEL_DIR = 'models'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EARLY_STOPPING_PATIENCE = 10

# GTZAN genres
GENRES = ['blues','classical','country','disco','hiphop',
          'jazz','metal','pop','reggae','rock']

class MLPNet(nn.Module):
    """Feed-forward network for handcrafted features."""
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.net(x)

class ConvNet(nn.Module):
    """Simple CNN for log-mel spectrograms."""
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Linear(64, num_classes)
    def forward(self, x):
        x = self.features(x).view(x.size(0), -1)
        return self.classifier(x)


def load_mlp_data(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Feature file not found: {path}")
    data = np.load(path, allow_pickle=True).item()
    return data['features'], data['labels']

def load_cnn_data(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Spectrogram file not found: {path}")
    data = np.load(path)
    return data['specs'], data['labels']


def train_mlp(data_path: str):
    X, y = load_mlp_data(data_path)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))

    train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train))
    val_ds   = TensorDataset(torch.from_numpy(X_val).float(),   torch.from_numpy(y_val))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y))
    model = MLPNet(input_dim, num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=5, factor=0.5, verbose=False
    )

    best_acc = 0.0
    epochs_no_improve = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        avg_loss = total_loss / len(train_loader.dataset)

        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                out = model(xb)
                y_pred.extend(torch.argmax(out, 1).cpu().tolist())
                y_true.extend(yb.cpu().tolist())
        val_acc = accuracy_score(y_true, y_pred)
        print(f"MLP Epoch {epoch}/{EPOCHS} — Loss: {avg_loss:.4f} — Val Acc: {val_acc:.4f}")
        scheduler.step(val_acc)
        if val_acc > best_acc:
            best_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'mlp_best.pth'))
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"⚠️ Early stopping MLP: no improvement for {EARLY_STOPPING_PATIENCE} epochs")
                break

    torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'mlp_last.pth'))
    print(f"✅ MLP training complete. Best Val Acc: {best_acc:.4f}")


def train_cnn(data_path: str):
    X, y = load_cnn_data(data_path)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train))
    val_ds   = TensorDataset(torch.from_numpy(X_val).float(),   torch.from_numpy(y_val))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    num_classes = len(np.unique(y))
    model = ConvNet(num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_acc = 0.0
    os.makedirs(MODEL_DIR, exist_ok=True)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                out = model(xb)
                y_pred.extend(torch.argmax(out, 1).cpu().tolist())
                y_true.extend(yb.cpu().tolist())
        val_acc = accuracy_score(y_true, y_pred)
        print(f"CNN Epoch {epoch}/{EPOCHS} — Val Acc: {val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'cnn_best.pth'))

    torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'cnn_last.pth'))
    print(f"✅ CNN training complete. Best Val Acc: {best_acc:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train genre classifier (MLP or CNN)")
    parser.add_argument('--model-type', choices=['mlp','cnn'], required=True,
                        help='Choose mlp for handcrafted features or cnn for spectrograms')
    parser.add_argument('--data', '-d', required=True,
                        help='Path to features (.npy) or specs (.npz) file')
    args = parser.parse_args()
    if args.model_type == 'mlp':
        train_mlp(args.data)
    else:
        train_cnn(args.data)