import torch
import torch.nn as nn

class Discriminator(nn.Module):
    """Conditional discriminator for face generation."""
    
    def __init__(self, embedding_dim=512, channels=3):
        super().__init__()
        
        # Project embedding
        self.embedding_proj = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.LeakyReLU(0.2)
        )
        
        # Initial block
        self.initial = nn.Sequential(
            nn.Conv2d(channels, 32, 4, 2, 1),  # 128x128 -> 64x64
            nn.LeakyReLU(0.2)
        )
        
        # Downsample blocks
        self.downsample = nn.ModuleList([
            self._make_block(32, 64),       # 64x64 -> 32x32
            self._make_block(64, 128),      # 32x32 -> 16x16
            self._make_block(128, 256),     # 16x16 -> 8x8
            self._make_block(256, 512),     # 8x8 -> 4x4
        ])
        
        # Final layers
        self.final = nn.Sequential(
            nn.Conv2d(512 + 512, 512, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, 4, 1, 0),  # 4x4 -> 1x1
            nn.Sigmoid()
        )
        
    def _make_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, x, embedding):
        """
        Forward pass of the discriminator.
        
        Args:
            x: image tensor of shape (batch_size, channels, 128, 128)
            embedding: face embedding tensor of shape (batch_size, embedding_dim)
            
        Returns:
            output: probability tensor of shape (batch_size, 1)
        """
        # Project embedding
        batch_size = x.size(0)
        emb = self.embedding_proj(embedding)
        emb = emb.view(batch_size, -1, 1, 1)
        emb = emb.expand(-1, -1, 4, 4)  # Expand to 4x4 spatial dimensions
        
        # Process image
        x = self.initial(x)
        
        # Apply downsampling blocks
        for block in self.downsample:
            x = block(x)
        
        # Concatenate embedding with features
        x = torch.cat([x, emb], dim=1)
        
        # Final layers
        x = self.final(x)
        return x.view(batch_size, -1) 