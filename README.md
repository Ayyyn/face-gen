# Face-Conditional Generative Model

A PyTorch-based generative model that generates 128x128 RGB images of human faces, conditioned on face embeddings from ArcFace.

## Features

- Conditional GAN architecture for face generation
- Uses ArcFace embeddings as conditioning input
- Efficient training pipeline with Weights & Biases integration
- Zero-shot generalization to unseen face embeddings
- Comprehensive metrics tracking (FID, L1 loss, etc.)

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/face-gen.git
cd face-gen
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the CelebA dataset and precompute embeddings:
```bash
python scripts/prepare_data.py
```

## Project Structure

```
face-gen/
├── models/              # Model architectures
│   ├── generator.py     # Generator network
│   ├── discriminator.py # Discriminator network
│   └── encoder.py       # Face embedding encoder
├── data/               # Dataset and data loading
│   ├── dataset.py      # Custom dataset class
│   └── transforms.py   # Data augmentation
├── training/           # Training utilities
│   ├── trainer.py      # Training loop
│   └── metrics.py      # Evaluation metrics
├── scripts/            # Utility scripts
│   ├── prepare_data.py # Data preparation
│   └── train.py        # Training script
├── notebooks/          # Jupyter notebooks
│   └── inference.ipynb # Inference notebook
└── configs/            # Configuration files
    └── default.yaml    # Default training config
```

## Training

To start training:

```bash
python scripts/train.py --config configs/default.yaml
```

Training progress can be monitored on the Weights & Biases dashboard.

## Inference

Use the provided inference notebook to generate faces from embeddings:

```bash
jupyter notebook notebooks/inference.ipynb
```

## Metrics

The model tracks the following metrics:
- L1 Loss
- FID Score
- Generator Loss
- Discriminator Loss
- Real/Fake Accuracy

## License

MIT License 