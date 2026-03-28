from torch import nn

class ResBlock(nn.Module):
    def __init__(self, channel: int = 64):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(channel, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(channel, affine=True),
        )

    def forward(self, x):
        return self.block(x) + x