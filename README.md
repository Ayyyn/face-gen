# Face-Conditional Generative Model

A PyTorch-based generative model that generates 128x128 RGB images of human faces, conditioned on face embeddings from ArcFace.

## Features

- Conditional GAN architecture for face generation
- Uses ArcFace embeddings as conditioning input
- Efficient training pipeline with Weights & Biases integration
- Zero-shot generalization to unseen face embeddings(scope of improvement)

## Setup

1. Clone the repository:
```bash
git clone https://github.com/Ayyyn/face-gen.git
cd face-gen
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the [ffhq-128*128_dataset](https://www.kaggle.com/datasets/greatgamedota/ffhq-face-data-set) and precompute embeddings using ArcFace
```bash
python3 prepare_data.py --data_dir <input_images_dir> --output_dir <output_embeddings_dir>
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
├── training/           # Training utilities
│   ├── trainer.py      # Training loop
│   └── metrics.py      # Evaluation metrics
├── scripts/            # Utility scripts
│   ├── prepare_data.py # Data preparation
│   └── train.py        # Training script
├──configs/             # Configuration files
|   └── default.yaml    # Default training config
├── inference.py        # Inference script
└── requirements.txt    # Required packages

```

## Training

To start training:

```bash
python scripts/train.py --image-dir <image_dir> --embedding_dir <embedding_dir> --output_dir <output_dir> --config configs/default.yaml
```
or
```bash
python scripts/train.py \
    --image_dir <image_dir> \
    --embedding_dir <embedding_dir> \
    --output_dir <output_dir> \
    --batch_size 32 \
    --num_epochs 100 \
    --lr 0.0002 \
    --lambda_l1 100.0
```

Training progress can be monitored on the Weights & Biases dashboard.

## Inference

Use the provided inference notebook to generate faces from embeddings:

```bash
python3 inference.py --embedding_path <embedding_file_path> --checkpoint_path <checkoint_path> --output_path <output_path>
```

## Metrics

The model tracks the following metrics:
- Discriminator Loss(d_loss)
- Total Generator Loss(g_loss)- sum of ```Generator's adversarial loss``` and ```Generator's L1 reconstruction loss```

## License

MIT License 
