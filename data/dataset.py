import os
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

class FaceDataset(Dataset):
    """Dataset class for face images and their embeddings."""
    
    def __init__(self, image_dir, embedding_dir, transform=None):
        """
        Args:
            image_dir (string): Directory with all the images
            embedding_dir (string): Directory with all the embeddings
            transform (callable, optional): Optional transform to be applied on images
        """
        self.image_dir = image_dir
        self.embedding_dir = embedding_dir
        
        # Get list of embedding files
        self.embedding_files = [f for f in os.listdir(embedding_dir) if f.endswith('.npy')]
        
        # Filter out embeddings that don't have corresponding images
        self.valid_files = []
        for emb_file in self.embedding_files:
            base_name = os.path.splitext(emb_file)[0]
            img_path = os.path.join(image_dir, f"{base_name}.png")  # Changed to .png
            if os.path.exists(img_path):
                self.valid_files.append(emb_file)
        
        print(f"Found {len(self.valid_files)} valid image-embedding pairs out of {len(self.embedding_files)} embeddings")
        
        if len(self.valid_files) == 0:
            raise ValueError("No valid image-embedding pairs found!")
        
        # Default transform if none provided
        self.transform = transform or A.Compose([
            A.Resize(128, 128),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ToTensorV2(),
        ])
    
    def __len__(self):
        return len(self.valid_files)
    
    def __getitem__(self, idx):
        # Get embedding
        emb_name = self.valid_files[idx]
        emb_path = os.path.join(self.embedding_dir, emb_name)
        embedding = torch.from_numpy(np.load(emb_path)).float()
        
        # Get corresponding image
        base_name = os.path.splitext(emb_name)[0]
        img_path = os.path.join(self.image_dir, f"{base_name}.png")  # Changed to .png
        
        # Load and transform image
        image = np.array(Image.open(img_path).convert('RGB'))
        if self.transform:
            image = self.transform(image=image)['image']
        
        return {
            'image': image,
            'embedding': embedding
        } 