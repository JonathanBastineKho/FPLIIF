import torch
import torch.nn.functional as F
from torch import nn
from src.components import RCMLP, PixelMLP

class Decoder(nn.Module):
    def __init__(self,
                 in_feat: int = 64,
                 class_num: int = 19,
                 out_size: int = 256):
        super().__init__()
        self.out_size = out_size
        self.rc_mlp = RCMLP(in_feat=in_feat * 9, 
                            hidden=256, 
                            out_feat=in_feat)
        
        pos_dim = 2
        self.pixel_mlp = PixelMLP(in_feat=in_feat*2+pos_dim,
                                  hidden=256,
                                  n_classes=class_num)
        
    def _make_pos_encoding(self, H: int, W: int, device) -> torch.Tensor:
        """
        Add [-1, 1] positional encoding
        """
        ys = torch.linspace(-1, 1, H, device=device)
        xs = torch.linspace(-1, 1, W, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        pos = torch.stack([grid_x, grid_y], dim=-1)
        return pos


    def _unfold_3x3(self, Z: torch.Tensor) -> torch.Tensor:
        B, C, H, W = Z.shape
        Z_padded = F.pad(Z, [1, 1, 1, 1], mode='replicate')
        patches = []
        for dy in range(3):
            for dx in range(3):
                patches.append(Z_padded[:, :, dy:dy+H, dx:dx+W])
        return torch.cat(patches, dim=1)
    
    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        B, C, _, _ = Z.shape
        out_H = out_W = self.out_size

        # Global features
        g = Z.mean(dim=[2, 3])

        # Local interpretation
        Z_unfolded = self._unfold_3x3(Z)
        Z_unfolded = Z_unfolded.permute(0, 2, 3, 1)
        Z_hat = self.rc_mlp(Z_unfolded)
        Z_hat = Z_hat.permute(0, 3, 1, 2)

        Z_up = F.interpolate(Z_hat, size=(out_H, out_W),
                             mode='bilinear', align_corners=False)
        Z_up = Z_up.permute(0, 2, 3, 1)

        g_expanded = g[:, None, None, :].expand(B, out_H, out_W, C)

        pos_enc = self._make_pos_encoding(out_H, out_W, Z.device)
        pos_enc = pos_enc.unsqueeze(0).expand(B, -1, -1, -1)

        pixel_input = torch.cat([Z_up, g_expanded, pos_enc], dim=-1)

        logits = self.pixel_mlp(pixel_input)
        logits = logits.permute(0, 3, 1, 2)
        return logits 