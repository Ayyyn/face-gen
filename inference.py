import os
import sys
import torch
import argparse
import numpy as np
from PIL import Image
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).resolve().parent.parent)
sys.path.append(project_root)

from models.generator import Generator
import torchvision.transforms as transforms
from torchvision.utils import save_image

def parse_args():
    parser = argparse.ArgumentParser(description='Generate faces from embeddings')
    parser.add_argument('--embedding_path', type=str, required=True,
                      help='Path to the embedding file (.npy)')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                      help='Path to the generator checkpoint')
    parser.add_argument('--output_path', type=str, default='generated_face.png',
                      help='Path to save the generated image')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to use (cuda/cpu)')
    args = parser.parse_args()
    
    # Convert paths to absolute paths
    args.embedding_path = str(Path(args.embedding_path).resolve())
    args.checkpoint_path = str(Path(args.checkpoint_path).resolve())
    args.output_path = str(Path(args.output_path).resolve())
    
    return args

def load_model(checkpoint_path, device):
    print(f"Loading model from {checkpoint_path}")
    try:
        # Initialize generator
        generator = Generator().to(device)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        print("Checkpoint contents:", checkpoint.keys())
        
        if 'generator' not in checkpoint:
            raise ValueError("Could not find generator state dict in checkpoint")
        
        generator.load_state_dict(checkpoint['generator'])
        generator.eval()
        print("Model loaded successfully")
        return generator
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

def generate_face(generator, embedding, device):
    print("Generating face...")
    try:
        # Generate random noise
        noise = torch.randn(1, 100, device=device)
        
        # Generate image
        with torch.no_grad():
            generated = generator(embedding, noise)
        print(f"Generated tensor shape: {generated.shape}")
        
        # Convert to PIL image
        generated = generated.squeeze(0).cpu()
        generated = (generated + 1) / 2  # Scale from [-1,1] to [0,1]
        print("Face generated successfully")
        return generated
    except Exception as e:
        print(f"Error generating face: {str(e)}")
        raise

def main():
    args = parse_args()
    print(f"Using device: {args.device}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    try:
        # Load embedding
        print(f"Loading embedding from {args.embedding_path}")
        embedding = torch.from_numpy(np.load(args.embedding_path)).float().to(args.device)
        embedding = embedding.unsqueeze(0)  # Add batch dimension
        print(f"Embedding shape: {embedding.shape}")
        
        # Load model
        generator = load_model(args.checkpoint_path, args.device)
        
        # Generate face
        generated = generate_face(generator, embedding, args.device)
        
        # Save image
        print(f"Saving image to {args.output_path}")
        save_image(generated, args.output_path)
        print(f"Generated face saved to {args.output_path}")
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main() 