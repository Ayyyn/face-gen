# Face-Conditional Generative Model

A PyTorch-based generative model that generates 128x128 RGB images of human faces, conditioned on face embeddings from ArcFace.

## Features

- Conditional GAN architecture for face generation
- Uses ArcFace embeddings as conditioning input
- Efficient training pipeline with Weights & Biases integration
- Zero-shot generalization to unseen face embeddings
  
## Scope of Improvement

- Hyperparameter Tuning
- Larger Dataset (Model uses 10k images only)
- Longer training period (current training involved only 200 epochs)
- Addition of metrics- FID and Inception Scores
- deeper generator
- check performance on decrease in noise usage

[wandb workspace url](https://wandb.ai/ayushinanavati/face-gen/runs/5d8mqezt?nw=nwuserayushinanavati)

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

3. Download the [ffhq-128*128_dataset](https://www.kaggle.com/datasets/greatgamedota/ffhq-face-data-set) and precompute embeddings using ArcFace(```buffalo_sc``` model was selected as it's light-weight)
```bash
python3 prepare_data.py --data_dir <input_images_dir> --output_dir <output_embeddings_dir>
```

- link to 70k images- [all_images](https://drive.google.com/file/d/1KHkdHwKRxWRYV_tRD-8ZY1q-6nUojjOE/view?usp=sharing)
- link to corresponding 70k embeddings- [all_embeddings](https://drive.google.com/drive/folders/1EtQHksQ9rS9m9VQNOTRe8O3f8B2o3Q-O?usp=drive_link)

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

### Resources

The model was trained for 200 epochs on T4 on Colab for approximately 1.5 hours

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

### Checkpoint Files

Checkpoint files at- [checkpoint_files](https://drive.google.com/file/d/14sJAtSQD9sBHN3kPfN-01WwMY8tPK7mI/view?usp=drive_link)

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
