from torch import nn
from src.models.encoder import Encoder
from src.models.decoder import Decoder

class FPLIIF(nn.Module):
    def __init__(self,
                 channel: int = 64,
                 class_num: int = 19,
                 out_size: int = 256):
        super().__init__()
        self.encoder = Encoder(channel=channel)
        self.decoder = Decoder(in_feat=channel,
                               class_num=class_num,
                               out_size=out_size)

    def forward(self, x):
        Z = self.encoder(x)
        return self.decoder(Z)
