# adapted from https://github.com/yitu-opensource/T2T-ViT
import numpy as np
import torch
import torch.nn as nn
from timm.models.layers import DropPath

from .attention import AxialAttention
from .attention import Attention as QKVConvAttention


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, in_dim=None, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.in_dim = in_dim
        # bug in T2T official code
        # in https://github.com/yitu-opensource/T2T-ViT/commit/097249357c177d0329e23fee981763897070890e
        # head_dim = dim // num_heads
        # should change to: head_dim = in_dim // num_heads
        head_dim = in_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, in_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(in_dim, in_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.in_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.in_dim)
        x = self.proj(x)
        x = self.proj_drop(x)

        # skip connection
        x = v.squeeze(1) + x  # because the original x has different size with current x, use v to do skip connection

        return x

    @staticmethod
    def compute_macs(module, input, output):
        # T: num_token
        # S: num_token
        input = input[0]
        _, T, C = input.shape
        C = module.in_dim
        S = T
        macs = 0
        n_params = 0

        # Scaled-dot-product macs
        # [B x T x C] x [B x C x S] --> [B x T x S]
        # multiplication-addition is counted as 1 because operations can be fused
        num_macs_kq = T * S * C
        # [B x T x S] x [B x S x C] --> [B x T x C]
        num_macs_v = T * C * S

        macs += num_macs_kq + num_macs_v
        # print('macs att', macs / 1e8)

        # self attention: T should be equal to S
        assert T == S
        qkv_params = sum([p.numel() for p in module.qkv.parameters()])
        n_params += qkv_params
        # multiply by Seq length
        macs += qkv_params * T
        # print('macs qkv', qkv_params * T / 1e8)

        proj_params = sum([p.numel() for p in module.proj.parameters()])
        n_params += proj_params
        macs += (proj_params * T)
        # print('macs proj', proj_params * T / 1e8)

        module.__flops__ += macs
        # return n_params, macs


class Token_transformer(nn.Module):

    def __init__(self, dim, in_dim, num_heads, mlp_ratio=1., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_axial_attn=False,
                 use_qkvattention=False, with_norm1=True, qkvconv_method='conv',
                 kernel_size=1, stride=1, padding=0):
        super().__init__()

        if with_norm1:
            self.norm1 = norm_layer(dim)
        else:
            self.norm1 = nn.Identity()

        self.dim_change = (dim != in_dim) or (stride>1)

        self.use_axial_attn = use_axial_attn
        self.use_qkvattention = use_qkvattention

        if self.use_axial_attn:
            self.attn_h = AxialAttention(
                in_dim, in_dim=dim, num_heads=num_heads,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop, use_cls_token=False,
                horizontal=True, use_full_attn_for_cls=True,
                with_proj=True, add_v_res=True
            )
            self.norm_axial = norm_layer(in_dim)
            self.attn_w = AxialAttention(
                in_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop, use_cls_token=False,
                horizontal=False, use_full_attn_for_cls=True,
                with_proj=True
            )
        else:
            if use_qkvattention:
                self.attn = QKVConvAttention(
                    in_dim, in_dim=dim, num_heads=num_heads,
                    qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                    use_qkvconv=True,
                    proj_drop=drop, kernel_size=kernel_size, stride=stride,
                    padding=padding, add_v_res=self.dim_change,
                    method=qkvconv_method
                )
            else:
                self.attn = Attention(
                    dim, in_dim=in_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                    proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(in_dim)

        self.mlp = Mlp(in_features=in_dim, hidden_features=int(in_dim * mlp_ratio), out_features=in_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x, x_cls=None):
        if x_cls is not None:
            C_cls = x_cls.shape[-1]
            C = x.shape[-1]
            if C != C_cls:
                x_cls = x_cls.repeat(1, 1, C // C_cls)
            x = torch.cat((x_cls, x), 1)
        res_x = x
        x = self.norm1(x)
        if self.use_axial_attn:
            attn_h, _prev = self.attn_h(x, None)
            attn, _prev = self.attn_w(self.norm_axial(attn_h), None)
            x = attn_h + self.drop_path(attn)
        else:
            x = self.attn(x)
            import sys
            print(x)
            sys.exit(-1)
            if self.use_qkvattention:
                # discard useless return 'prev'
                x = x[0]
            if self.use_qkvattention and not self.dim_change:
                # print('res_x', res_x.shape, x.shape)
                x = res_x + self.drop_path(x)
        # if x_cls is not None:
        #     x = torch.cat((x_cls, x), 1)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class T2T_module(nn.Module):
    """
    Tokens-to-Token encoding module
    """

    def __init__(self, img_size=224, tokens_type='transformer', in_chans=3, embed_dim=768, token_dim=[64, 64],
                 use_axial_attn=False, qkvconv_method='conv', depth=1, mlp_ratio=1.0, num_heads=1, with_bn=False):
        super().__init__()
        self.method = tokens_type
        self.bn = None
        if tokens_type == 'transformer':
            # transformer -> transformer -> convolution
            print('adopt transformer encoder for tokens-to-token')
            self.soft_split0 = nn.Unfold(kernel_size=(7, 7), stride=(4, 4), padding=(2, 2))
            self.soft_split1 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            self.soft_split2 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

            self.attention1 = Token_transformer(dim=in_chans * 7 * 7, in_dim=token_dim[1], num_heads=num_heads,
                                                mlp_ratio=mlp_ratio,
                                                use_axial_attn=use_axial_attn)
            self.attention2 = Token_transformer(dim=token_dim[0] * 3 * 3, in_dim=token_dim[1], num_heads=num_heads,
                                                mlp_ratio=mlp_ratio,
                                                use_axial_attn=use_axial_attn
                                                )
            self.project = nn.Linear(token_dim[1] * 3 * 3, embed_dim)

        elif tokens_type == 'qkvconv_transformer':

            print('adopt qkv conv transformer encoder for tokens-to-token')
            # self.soft_split0 = nn.Unfold(kernel_size=(7, 7), stride=(4, 4), padding=(2, 2))
            # self.soft_split1 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

            # for first layer, dim 3 -> 64, use regular convolution
            self.attention1 = Token_transformer(dim=in_chans, in_dim=token_dim[0], num_heads=num_heads,
                                                mlp_ratio=mlp_ratio,
                                                use_axial_attn=use_axial_attn, kernel_size=7, stride=4, padding=2,
                                                use_qkvattention=True, with_norm1=False, qkvconv_method='conv'
                                                )

            if depth > 1:
                self.attention2 = nn.Sequential(
                    *[Token_transformer(dim=token_dim[0] if _i == 0 else token_dim[1], in_dim=token_dim[1],
                                        num_heads=num_heads, mlp_ratio=mlp_ratio,
                                        use_axial_attn=use_axial_attn, kernel_size=3,
                                        stride=2 if _i == 0 else 1,
                                        padding=1,
                                        use_qkvattention=True, qkvconv_method=qkvconv_method
                                        )
                      for _i in range(depth)]
                )

            else:
                self.attention2 = Token_transformer(dim=token_dim[0], in_dim=token_dim[1], num_heads=num_heads,
                                                    mlp_ratio=mlp_ratio,
                                                    use_axial_attn=use_axial_attn, kernel_size=3, stride=2, padding=1,
                                                    use_qkvattention=True, qkvconv_method=qkvconv_method
                                                    )


            self.soft_split2 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            self.project = nn.Linear(token_dim[1] * 3 * 3, embed_dim)

        elif tokens_type == 'convolution':  # just for comparison with conolution, not our model
            # convolution -> convolution -> convolution
            # for this tokens type, you need change forward as three convolution operation
            print('adopt convolution layers for tokens-to-token')
            self.soft_split0 = nn.Conv2d(3, token_dim, kernel_size=(7, 7), stride=(4, 4),
                                         padding=(2, 2))  # the 1st convolution
            self.soft_split1 = nn.Conv2d(token_dim, token_dim, kernel_size=(3, 3), stride=(2, 2),
                                         padding=(1, 1))  # the 2nd convolution
            self.project = nn.Conv2d(token_dim, embed_dim, kernel_size=(3, 3), stride=(2, 2),
                                     padding=(1, 1))  # the 3rd convolution
        elif tokens_type == 'hybrid':
            # convolution -> transformer -> convolution
            print('adopt transformer encoder for tokens-to-token')
            self.soft_split0 = nn.Conv2d(3, token_dim, kernel_size=(7, 7), stride=(4, 4),
                                         padding=(2, 2))  # the 1st convolution
            self.soft_split1 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            self.soft_split2 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            self.attention2 = Token_transformer(dim=token_dim * 3 * 3, in_dim=token_dim, num_heads=num_heads,
                                                mlp_ratio=mlp_ratio)
            self.project = nn.Linear(token_dim * 3 * 3, embed_dim)
        elif tokens_type == 'hybrid_qkvconv_transformer':
            self.soft_split0 = nn.Conv2d(3, token_dim[0], kernel_size=(7, 7), stride=(4, 4),
                                         padding=(2, 2))  # the 1st convolution
            if with_bn:
                self.bn = nn.BatchNorm2d(token_dim[0])

            if depth > 1:
                self.attention2 = nn.Sequential(
                    *[
                        Token_transformer(
                            dim=token_dim[0] if _i == 0 else token_dim[1],
                            in_dim=token_dim[1],
                            num_heads=num_heads, mlp_ratio=mlp_ratio,
                            use_axial_attn=use_axial_attn, kernel_size=3,
                            stride=2 if _i == 0 else 1,
                            padding=1,
                            use_qkvattention=True,
                            qkvconv_method=qkvconv_method
                        ) for _i in range(depth)
                    ]
                )

            else:
                self.attention2 = Token_transformer(
                    dim=token_dim[0], in_dim=token_dim[1], num_heads=num_heads,
                    mlp_ratio=mlp_ratio, use_axial_attn=use_axial_attn,
                    kernel_size=3, stride=2, padding=1,
                    use_qkvattention=True, qkvconv_method=qkvconv_method
                )

            self.soft_split2 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            self.project = nn.Linear(token_dim[1] * 3 * 3, embed_dim)

        else:
            assert False, 'T2T type {} not supported'.format(tokens_type)

        self.num_patches = (img_size // (4 * 2 * 2)) * (
                    img_size // (4 * 2 * 2))  # there are 3 sfot split, stride are 4,2,2 seperately

    def forward_transformer(self, x):
        # step0: soft split
        if self.method == 'transformer':
            x = self.soft_split0(x).transpose(1, 2)
            # iteration1: restricturization/reconstruction
            x = self.attention1(x)
            B, new_HW, C = x.shape
            x = x.transpose(1, 2).view(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        elif self.method == 'hybrid':
            x = self.soft_split0(x)
        else:
            assert False, 'T2T method {} not supported in forward_transformer'.format(self.method)

        # iteration1: soft split
        x = self.soft_split1(x).transpose(1, 2)

        # iteration2: restricturization/reconstruction
        x = self.attention2(x)
        B, new_HW, C = x.shape
        x = x.transpose(1, 2).view(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        # iteration2: soft split
        x = self.soft_split2(x).transpose(1, 2)

        # final tokens
        x = self.project(x)

        return x

    def forward_conv(self, x):
        # step0: soft split
        x = self.soft_split0(x)
        x = self.soft_split1(x)
        # final tokens
        x = self.project(x).flatten(-2, -1).transpose(1, 2)

        return x

    def forward_qkvconv_transformer(self, x):
        if self.method == 'hybrid_qkvconv_transformer':
            x = self.soft_split0(x)
            if self.bn is not None:
                x = self.bn(x)
            x = x.flatten(2).transpose(1, 2)
        else:
            x = x.flatten(2).transpose(1, 2)
            x = self.attention1(x)

        x = self.attention2(x)

        B, new_HW, C = x.shape
        x = x.transpose(1, 2).view(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))

        # iteration2: soft split
        x = self.soft_split2(x).transpose(1, 2)

        # final tokens
        x = self.project(x)

        return x

    def forward(self, x):
        if self.method == 'transformer' or self.method == 'hybrid':
            return self.forward_transformer(x)
        elif self.method == 'qkvconv_transformer' or self.method == 'hybrid_qkvconv_transformer':
            return self.forward_qkvconv_transformer(x)
        else:
            return self.forward_conv(x)


class DownsampleToken(nn.Module):
    """
    Tokens-to-Token encoding module
    """
    def __init__(self,
                 in_channels,
                 kernel_size,
                 stride=1,
                 padding=1,
                 method='transformer',
                 num_heads=1,
                 mlp_ratio=1,
                 use_cls_token=True
                 ):
        super(DownsampleToken, self).__init__()
        # does not change feature dim for now
        self.method = method
        self.use_cls_token = use_cls_token
        if method == 'transformer':
            self.soft_split = nn.Unfold(kernel_size=kernel_size, stride=stride, padding=padding)
            self.attn = Token_transformer(dim=in_channels * kernel_size * kernel_size,
                                          in_dim=in_channels, num_heads=num_heads, mlp_ratio=mlp_ratio)

        elif method == 'convolution':  # just for comparison with conolution, not our model
            # for this tokens type, you need change forward as three convolution operation
            self.soft_split = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                        stride=stride, padding=padding)  # the 1st convolution
            self.attn = Token_transformer(dim=in_channels,
                                          in_dim=in_channels, num_heads=num_heads, mlp_ratio=mlp_ratio)


    def forward(self, x, prev=None):
        # x: B x num_token x dim
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        if self.use_cls_token:
            x_cls, x = torch.split(x, [1, N-1], 1)
            x_cls = x_cls.view(B, 1, C)

        # B x C x H x W
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)

        if self.method == 'convolution':
            x = self.soft_split(x).flatten(-2, -1).transpose(1, 2)
        else:
            # unfold: nchw -> nc'L
            # B x C' x L
            x = self.soft_split(x).transpose(1, 2)

        # iteration1: restricturization/reconstruction
        x = self.attn(x, x_cls)

        return x, prev

class UpsampleToken(nn.Module):
    """
    Tokens-to-Token encoding module
    """
    def __init__(self,
                 in_channels,
                 kernel_size=3,
                 stride=2,
                 padding=2,
                 method='transformer',
                 num_heads=1,
                 mlp_ratio=1,
                 use_cls_token=True
                 ):
        super(UpsampleToken, self).__init__()
        # does not change feature dim for now
        self.method = method
        self.use_cls_token = use_cls_token
        if method == 'transformer':
            self.up = torch.nn.Upsample(scale_factor=2, mode='bilinear')
            self.attn = Token_transformer(dim=in_channels,
                                          in_dim=in_channels, num_heads=num_heads, mlp_ratio=mlp_ratio)

        elif method == 'convolution':  # just for comparison with conolution, not our model
            # for this tokens type, you need change forward as three convolution operation
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding)  # the 1st convolution
            self.attn = Token_transformer(dim=in_channels,
                                          in_dim=in_channels, num_heads=num_heads, mlp_ratio=mlp_ratio)


    def forward(self, x, prev=None):
        # x: B x num_token x dim
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        if self.use_cls_token:
            x_cls, x = torch.split(x, [1, N-1], 1)
            x_cls = x_cls.view(B, 1, C)
        else:
            x_cls = None

        # B x C x H x W
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)

        x = self.up(x).flatten(-2, -1).transpose(1, 2)

        # iteration1: restricturization/reconstruction
        x = self.attn(x, x_cls)

        return x, prev
