import torch.nn as nn
from .cross_attn import cross_spatial_qk2v_k_relu
from timm.models.layers import Mlp
from .mid_layer_registry import fusion_registry


@fusion_registry.register()
class SCAM(nn.Module):
    def __init__(self, dim=768, reduction=1, num_heads=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.cross_attn = cross_spatial_qk2v_k_relu(dim // reduction, num_heads=num_heads)

        self.end_proj1 = nn.Linear(dim, dim)
        self.end_proj2 = nn.Linear(dim, dim)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        self.ffn1 = Mlp(dim, dim)

    def forward(self, x1, x2):
        v1, v2 = self.cross_attn(x1, x2)

        y1 = x1 + v1
        y2 = x2 + v2
        out_x1 = x1 + self.norm1(self.end_proj1(y1))
        out_x2 = x2 + self.norm2(self.end_proj2(y2))

        out_x1 = x1 + self.ffn1(out_x1)
        out_x2 = x2 + self.ffn1(out_x2)
        return out_x1, out_x2
