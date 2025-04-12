import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    """Lightweight conditional GAN generator for face generation."""
    
    def __init__(self, embedding_dim=512, noise_dim=100, channels=3):
        super().__init__()
        
        # Project embedding and noise to higher dimension
        self.embedding_proj = nn.Linear(embedding_dim, 256)
        self.noise_proj = nn.Linear(noise_dim, 256)
        
        # Initial dense layer
        self.dense = nn.Sequential(
            nn.Linear(512, 512 * 4 * 4),
            nn.ReLU()
        )
        
        # Upsample blocks
        self.upsample = nn.ModuleList([
            self._make_block(512, 256),     # 4x4 -> 8x8
            self._make_block(256, 128),     # 8x8 -> 16x16
            self._make_block(128, 64),      # 16x16 -> 32x32
            self._make_block(64, 32),       # 32x32 -> 64x64
            self._make_block(32, 32)        # 64x64 -> 128x128
        ])
        
        # Final layer
        self.final = nn.Sequential(
            nn.Conv2d(32, channels, 3, 1, 1),
            nn.Tanh()
        )
        
    def _make_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, embedding, noise):
        """
        Forward pass of the generator.
        
        Args:
            embedding: face embedding tensor of shape (batch_size, embedding_dim)
            noise: random noise tensor of shape (batch_size, noise_dim)
            
        Returns:
            generated_image: tensor of shape (batch_size, channels, 128, 128)
        """
        # Project inputs
        batch_size = embedding.size(0)
        emb = self.embedding_proj(embedding)
        z = self.noise_proj(noise)
        
        # Combine embedding and noise
        x = torch.cat([emb, z], dim=1)
        
        # Initial dense layer
        x = self.dense(x)
        x = x.view(batch_size, 512, 4, 4)
        
        # Apply upsample blocks
        for block in self.upsample:
            x = block(x)
            
        # Final layer
        x = self.final(x)
        return x 