import torch
from torch import nn


class SRSelfAttention(nn.Module):
    def __init__(self, dim, rratio=2, num_heads=8, qkv_bias=False,
                 qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert (dim % num_heads) == 0, 'dimension must be divisible by the number of heads'

        self.rratio = rratio

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5

        self.query = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.proj_sr = nn.Conv2d(dim, dim, kernel_size=rratio, stride=rratio,
                                 bias=False)
        self.norm = nn.InstanceNorm2d(dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, nx=None, ny=None):
        b, n, d = x.shape
        d_h, h = self.head_dim, self.num_heads

        # get queries
        queries = self.scale * self.query(x).reshape(b, n, h, d_h).transpose(1, 2)

        # spatial reduction for k and v
        x_local = x[:, -nx * ny:].transpose(-2, -1).reshape(b, d, nx, ny)
        x_local = self.norm(self.proj_sr(x_local)).view(b, d, -1)
        x = torch.cat([x[:, :-nx * ny], x_local.transpose(-2, -1)], dim=1)
        # compute keys and values
        kv = self.kv(x).reshape(b, -1, 2, d).permute(2, 0, 3, 1)
        keys, values = kv[0], kv[
            1]  # make torchscript happy (cannot use tensor as tuple) b x d x k

        # merge head into batch for queries and key / values
        merge_key_values = lambda t: t.reshape(b, h, d_h, -1).transpose(-2, -1)
        keys, values = map(merge_key_values, (keys, values))  # b x h x k x d_h

        # attention
        attn = torch.einsum('bhnd,bhkd->bhnk', queries, keys)
        attn = (attn - torch.max(attn, dim=-1, keepdim=True)[0]).softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = torch.einsum('bhnk,bhkd->bhnd', attn, values)

        # split heads
        out = out.transpose(1, 2).reshape(b, n, -1)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out
