import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from models.encoder import FaceEncoder

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Path to save embeddings')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of workers for data loading')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize encoder
    encoder = FaceEncoder()
    
    # Get list of image files
    image_files = [f for f in os.listdir(args.data_dir) 
                  if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    # Process images
    for img_name in tqdm(image_files, desc='Processing images'):
        # Load image
        img_path = os.path.join(args.data_dir, img_name)
        image = np.array(Image.open(img_path).convert('RGB'))
        
        try:
            # Get embedding
            embedding = encoder.get_embedding(image)
            
            # Save embedding
            emb_name = os.path.splitext(img_name)[0] + '.npy'
            emb_path = os.path.join(args.output_dir, emb_name)
            np.save(emb_path, embedding.cpu().numpy())
            
        except Exception as e:
            print(f"Error processing {img_name}: {str(e)}")
            continue

if __name__ == '__main__':
    main() 