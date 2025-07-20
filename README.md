# AI Guitar Pedal

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE) [![Python Version](https://img.shields.io/badge/Python-3.7%2B-lightgrey.svg)](https://www.python.org/)

**Real‑time, genre‑aware guitar effects processing powered by a lightweight CNN, fully deployable on a Raspberry Pi.**

---

## Table of Contents

1. [Features](#features)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Repository Structure](#repository-structure)
6. [Development](#development)
7. [Contributing](#contributing)
8. [License](#license)
9. [Contact](#contact)

---

## Features

* **Real‑time audio I/O**: Low‑latency capture and playback via USB audio interface or Pi DAC HAT.
* **Genre classification**: On‑the‑fly style detection with a pre‑trained convolutional neural network.
* **Effects chain**: Configurable modules for distortion, chorus, reverb, and more.
* **Optimized for Raspberry Pi**: Efficient processing to maintain <30 ms round‑trip latency.
* **Modular design**: Separate training, preprocessing, and live‑inference components.

---

## Prerequisites

* **Hardware**: Raspberry Pi 4 (4 GB or 8 GB), USB audio interface or compatible DAC HAT.
* **Operating System**: Raspberry Pi OS (64‑bit) Lite or Desktop.
* **Software**: Python 3.7 or later, Git.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/raghav-sharma-15/ai_guitar_pedal.git
cd ai_guitar_pedal
```

### 2. Set up a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** For ARM‑compatible PyTorch builds, refer to [pytorch-arm-builds](https://github.com/nmilosev/pytorch-arm-builds).

---

## Usage

### Live Mode (Raspberry Pi)

1. Connect instrument to the audio interface and amp/headphones to output.
2. Activate the virtual environment:

   ```bash
   source venv/bin/activate
   ```
3. Launch real‑time processing:

   ```bash
   python live_guitar_pedal.py
   ```

### Batch Mode (Offline Processing)

Process a WAV file and generate an effects‑processed output:

```bash
python main.py --mode demo --input path/to/input.wav
# Result saved as out_cnn.wav
```

---

## Repository Structure

```
AI_GUITAR_PEDAL/
├── data/                  # Sample audio and feature data (gitignored)
│   ├── gtzan/             # Raw dataset directory
│   ├── features.npy       # Extracted feature vectors
│   └── specs.npz          # Spectrogram arrays
├── models/                # Pre‑trained model weights (gitignored)
│   ├── cnn_best.pth
│   ├── cnn_last.pth
│   ├── genre_model_best.pth
│   ├── genre_model_last.pth
│   └── scaler.pkl         # Feature standardizer
├── scripts/               # Utilities
│   ├── preprocess_audio.py  # Feature extraction
│   ├── effects.py           # Effects implementations
│   └── train_model.py       # Training and evaluation scripts
├── live_guitar_pedal.py   # Entry point for live inference
├── main.py                # Batch processing and demo
├── requirements.txt       # Python dependencies
├── LICENSE                # License file
└── README.md              # Project overview
```

---

## Development

### Training a new model

1. Prepare data in `data/gtzan/`.
2. Extract features:

   ```bash
   python scripts/preprocess_audio.py --mode specgram
   ```
3. Train:

   ```bash
   python scripts/train_model.py --model cnn --data data/specs.npz
   ```
4. Check `models/` for updated `.pth` files.

### Customizing effects

Edit `scripts/effects.py` to add or tune effect modules. Each effect exposes a standard interface for chaining.

---

## Contributing

1. Fork the repository.
2. Create a feature branch:

   ```bash
   git checkout -b feat/your-feature-name
   ```
3. Commit your changes:

   ```bash
   git commit -m "Add <feature>"
   ```
4. Push and open a pull request:

   ```bash
   git push origin feat/your-feature-name
   ```

Please ensure code adheres to PEP 8 and include tests where applicable.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Contact

**Raghav Sharma**
GitHub: [@raghav-sharma-15](https://github.com/raghav-sharma-15)
