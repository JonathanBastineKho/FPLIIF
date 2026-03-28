from torch import nn

class RCMLP(nn.Module):
    def __init__(self, in_feat: int = 576, hidden: int = 256, out_feat: int = 64):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features=in_feat, out_features=hidden),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=hidden, out_features=out_feat),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)
    
class PixelMLP(nn.Module):
    def __init__(self, in_feat: int = 130,
                 hidden: int = 256,
                 n_classes: int = 19,
                 n_layers: int = 4):
        super().__init__()
        layers = []
        ch = in_feat

        for _ in range(n_layers):
            layers.append(nn.Linear(ch, hidden))
            layers.append(nn.ReLU(inplace=True))
            ch = hidden

        layers.append(nn.Linear(hidden, n_classes))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)