from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import typing as t


class TinyEncoder(nn.module):
    """Basic prototype for CNN encoding. Planning to use simple blocks.
    conv -> ReLU -> conv -> ReLU -> GAP -> Linear

    Args:
        nn (_type_): _description_
    """
    def __init__(
        self,
        in_channels: int = 6,                           # In channels
        embed_dim: int = 64,                            # Dimension of state embedding
        conv_channels: t.Tuple[int, int] = (16, 32),    # (C1, C2) filters 
        use_groupnorm: bool = False,                    # Whether to use group normalization
        groups: int = 6,                                # Number of groups (ignored if use_groupnorm is false
    ):
        
        super().__init__()
        
        c1 = conv_channels[0]
        c2 = conv_channels[1]
        
        
        # Block 1, small 3x3, stride 1, padding matches stride to preserve hallway 
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=c1, kernel_size=3, stride=1, padding=1, bias=not use_groupnorm),
            nn.GroupNorm(num_groups=min(groups, c1), num_channels=c1) if use_groupnorm else nn.Identity(),
            nn.ReLU()
            # Not using MaxPool to avoid erasing corridor geometry
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=3, stride=1, padding=1, bias=not use_groupnorm),
            nn.GroupNorm(num_groups=min(groups, c2), num_channels=c2) if use_groupnorm else nn.Identity(),
            nn.ReLU()
        )
        
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        
        self.proj = nn.Linear(c2, embed_dim)
        