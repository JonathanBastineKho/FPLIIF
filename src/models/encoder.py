from torch import nn
import torch.nn.functional as F
from src.components import ResBlock

class Encoder(nn.Module):
    def __init__(self, channel: int = 64):
        super().__init__()
        self.first_conv = nn.Conv2d(3, channel, kernel_size=3, padding=1, bias=False)

        self.blocks_1 = nn.Sequential(*[ResBlock(channel) for _ in range(2)])
        self.down_1   = nn.Conv2d(channel, channel, kernel_size=3, stride=2, padding=1, bias=False)
        self.blocks_2 = nn.Sequential(*[ResBlock(channel) for _ in range(6)])
        self.down_2   = nn.Conv2d(channel, channel, kernel_size=3, stride=2, padding=1, bias=False)
        self.blocks_3 = nn.Sequential(*[ResBlock(channel) for _ in range(16)])

    def forward(self, x):
        x = self.first_conv(x)
        
        x1 = self.down_1(self.blocks_1(x) + x)
        x2 = self.down_2(self.blocks_2(x1) + x1)

        Z = self.blocks_3(x2) + x2
        return Z    