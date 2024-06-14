import torch.nn as nn


class cross_spatial_qk2v_k_relu(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=False, qk_scale=None):
        super(cross_spatial_qk2v_k_relu, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.kv1 = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.kv2 = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.act = nn.ReLU()

    def forward(self, x1, x2):
        B, N, C = x1.shape
        q1 = x1.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        q2 = x2.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()

        k1, v1 = self.kv1(x1).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        k2, v2 = self.kv2(x2).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()

        ctx1 = (q1 @ k2.transpose(-2, -1)) * self.scale
        ctx1 = self.act(ctx1)

        ctx2 = (q2 @ k1.transpose(-2, -1)) * self.scale
        ctx2 = self.act(ctx2)

        x1 = (ctx1 @ v1).permute(0, 2, 1, 3).reshape(B, N, C).contiguous()
        x2 = (ctx2 @ v2).permute(0, 2, 1, 3).reshape(B, N, C).contiguous()

        return x1, x2
