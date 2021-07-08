import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def swish(x, inplace: bool = False):
    """
    Swish - Described in: https://arxiv.org/abs/1710.05941
    """
    return x.mul_(x.sigmoid()) if inplace else x.mul(x.sigmoid())


class Swish(nn.Module):
    def __init__(self, inplace: bool = False):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return swish(x, self.inplace)


class AxialAttention(nn.Module):
    def __init__(self, dim,
                 num_heads=8,
                 in_dim=None,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 use_cls_token=True,
                 horizontal=False,
                 use_full_attn_for_cls=False,
                 with_proj=True,
                 proj=None,
                 qkv=None,
                 add_v_res=False
                 ):
        super(AxialAttention, self).__init__()
        # handle change of feature dim
        if in_dim is None:
            in_dim = dim
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        if qkv is not None:
            self.qkv = qkv
        else:
            self.qkv = nn.Linear(in_dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.use_cls_token = use_cls_token
        self.horizontal = horizontal
        self.use_full_attn_for_cls = use_full_attn_for_cls and use_cls_token
        self.with_proj = with_proj
        if self.with_proj:
            self.proj = proj if proj is not None else nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(proj_drop)
        self.add_v_res = add_v_res


    def forward(self, x, prev=None):
        assert prev is None
        B, N, C = x.shape
        B_ori = B
        # 3 x B x num_heads x N x dim

        if self.use_cls_token:
            x_cls = x[:, 0]
            x = x[:, 1:]
        # TODO: for nonsquare input
        H = W = int(math.sqrt(N))
        x = x.reshape(B, H, W, C)
        if not self.horizontal:
            x = x.permute(0, 2, 1, 3)

        # if self.use_cls_token:
        #     x = torch.cat((x_cls.view(B, 1, 1, C).repeat(1, H, 1, 1), x), axis=2)  # B x H x (W+1) x C
        if self.use_cls_token:
            # B x (1+HW) x C
            x = torch.cat((x_cls.contiguous().view(B, 1, C), x.contiguous().view(B, H*W, C)), 1)
        else:
            x = x.contiguous().view(B, H*W, C)

        qkv = self.qkv(x) \
            .reshape(B, N, 3, self.num_heads, self.dim // self.num_heads) \
            .permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.add_v_res:
            v_res = v

        if self.use_cls_token:
            q_cls, q = torch.split(q, [1, N-1], -2)
            k_cls, k = torch.split(k, [1, N-1], -2)
            v_cls, v = torch.split(v, [1, N-1], -2)
            q = torch.cat((q_cls.view(B, self.num_heads, 1, 1, self.dim // self.num_heads)
                           .repeat(1, 1, H, 1, 1),
                           q.view(B, self.num_heads, H, W, self.dim // self.num_heads)),
                          -2).transpose(1, 2).flatten(0, 1)
            k = torch.cat((k_cls.view(B, self.num_heads, 1, 1, self.dim // self.num_heads)
                           .repeat(1, 1, H, 1, 1),
                           k.view(B, self.num_heads, H, W, self.dim // self.num_heads)),
                          -2).transpose(1, 2).flatten(0, 1)
            v = torch.cat((v_cls.view(B, self.num_heads, 1, 1, self.dim // self.num_heads)
                           .repeat(1, 1, H, 1, 1),
                           v.view(B, self.num_heads, H, W, self.dim // self.num_heads)),
                          -2).transpose(1, 2).flatten(0, 1)

        attn_score = (q @ k.transpose(-2, -1)) * self.scale

        if self.use_full_attn_for_cls:
            # get attention map for cls token
            attn_score_cls, attn_score = torch.split(attn_score, [1, W], dim=-2)
            attn_score_cls = attn_score_cls.reshape(B_ori, H, self.num_heads, 1 + W).transpose(1, 2)
            attn_score_cls = torch.cat((attn_score_cls[:, :, 0, 0:1],
                                        attn_score_cls[:, :, :, 1:].flatten(2, 3)), axis=2)
            attn_score_cls = F.softmax(attn_score_cls.unsqueeze(2), dim=-1)  # ori_B x num_heads x 1 x (1+HW)
            attn_cls = self.attn_drop(attn_score_cls)
            v_cls = v.reshape(B_ori, H, self.num_heads, W + 1, self.dim // self.num_heads).transpose(1, 2)
            v_cls = torch.cat((v_cls[:, :, 0, 0:1],
                               v_cls[:, :, :, 1:].flatten(2, 3)), axis=2)

            x_cls = (attn_cls @ v_cls).transpose(1, 2).reshape(B_ori, 1, self.dim)

        attn = F.softmax(attn_score, dim=-1)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N - 1 if self.use_full_attn_for_cls else N, self.dim)

        if self.use_cls_token and not self.use_full_attn_for_cls:
            x = x.reshape(B_ori, H, W + 1, self.dim)
            x_cls = x[:, :, 0].mean(1, keepdim=True)  # B x C
            x = x[:, :, 1:]
        else:
            x = x.reshape(B_ori, H, W, self.dim)
        if not self.horizontal:
            x = x.permute(0, 2, 1, 3)
        x = x.flatten(1, 2)
        if self.use_cls_token:
            x = torch.cat((x_cls, x), axis=1)

        if self.with_proj:
            x = self.proj(x)
            x = self.proj_drop(x)

        # add residual here as feature dimension change
        if self.add_v_res:
            x = v_res.transpose(1, 2).flatten(2) + x

        return x, prev

    @staticmethod
    def compute_macs(module, input, output):
        # T: num_token
        # S: num_token
        input = input[0]
        _, T, C = input.shape
        C = module.dim
        T_ori = T
        H = W = int(math.sqrt(T))
        macs = 0
        n_params = 0

        # Scaled-dot-product macs
        # [B x T x C] x [B x C x S] --> [B x T x S]
        # multiplication-addition is counted as 1 because operations can be fused
        num_macs_kq = H * (1+W) * (1+W) * C
        # [B x T x S] x [B x S x C] --> [B x T x C]
        num_macs_v = H * W * C * (1+W) + 1 * T_ori * C

        macs += num_macs_kq + num_macs_v
        # print('macs att', macs)

        # self attention: T should be equal to S
        assert W == H
        qkv_params = sum([p.numel() for p in module.qkv.parameters()])
        n_params += qkv_params
        # multiply by Seq length
        macs += qkv_params * T_ori
        # print('macs qkv', qkv_params * T_ori)

        proj_params = sum([p.numel() for p in module.proj.parameters()])
        n_params += proj_params
        macs += (proj_params * T_ori)
        # print('macs proj', proj_params * T_ori)

        module.__flops__ += macs


class CrissCrossAttention(nn.Module):
    def __init__(self, dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 use_cls_token=True,
                 proj=None,
                 qkv=None
                 ):
        super(CrissCrossAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias) if qkv is None else qkv
        self.attn_drop = nn.Dropout(attn_drop)
        self.use_cls_token = use_cls_token
        self.proj = proj if proj is not None else nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, prev=None):
        assert prev is None, 'residual attention not implemented'
        B, N, C = x.shape
        H = W = int(math.sqrt(N))

        qkv = self.qkv(x) \
            .reshape(B, N, 3, self.num_heads, C // self.num_heads) \
            .permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B x num_head x num_token x dim

        if self.use_cls_token:
            q_full, k_full, v_full = q, k, v
            # q_cls: B x num_head x dim
            q_cls, q = torch.split(q, [1, N-1], -2)
            k_cls, k = torch.split(k, [1, N-1], -2)
            v_cls, v = torch.split(v, [1, N-1], -2)

        # print(q.shape)
        q_h = q.view(B, self.num_heads, H, W, C // self.num_heads). \
            permute(0, 3, 1, 2, 4).contiguous().view(B * W, self.num_heads, H, C // self.num_heads)
        q_w = q.view(B, self.num_heads, H, W, C // self.num_heads).\
            permute(0, 2, 1, 3, 4).contiguous().view(B * H, self.num_heads, W, C // self.num_heads)

        k_h = k.view(B, self.num_heads, H, W, C // self.num_heads). \
            permute(0, 3, 1, 2, 4).contiguous().view(B * W, self.num_heads, H, C // self.num_heads)
        k_w = k.view(B, self.num_heads, H, W, C // self.num_heads). \
            permute(0, 2, 1, 3, 4).contiguous().view(B * H, self.num_heads, W, C // self.num_heads)

        v_h = v.view(B, self.num_heads, H, W, C // self.num_heads). \
            permute(0, 3, 1, 2, 4).contiguous().view(B * W, self.num_heads, H, C // self.num_heads)
        v_w = v.view(B, self.num_heads, H, W, C // self.num_heads). \
            permute(0, 2, 1, 3, 4).contiguous().view(B * H, self.num_heads, W, C // self.num_heads)

        # broadcast cls token in one direction
        if self.use_cls_token:
            # (B x W) x num_heads x (1+H) x dim
            q_h = torch.cat((q_cls.view(B, self.num_heads, 1, 1, C // self.num_heads)
                             .repeat(1, 1, W, 1, 1)
                             .permute(0, 2, 1, 3, 4).flatten(0, 1)
                             , q_h), -2)
            k_h = torch.cat((k_cls.view(B, self.num_heads, 1, 1, C // self.num_heads)
                             .repeat(1, 1, W, 1, 1)
                             .permute(0, 2, 1, 3, 4).flatten(0, 1)
                             , k_h), -2)
            v_h = torch.cat((v_cls.view(B, self.num_heads, 1, 1, C // self.num_heads)
                             .repeat(1, 1, W, 1, 1)
                             .permute(0, 2, 1, 3, 4).flatten(0, 1)
                             , v_h), -2)

        # (B * W) x num_heads x H x H, (B x W) x num_heads x (1+H) x (1+H) if with cls token
        attn_score_h = (q_h @ k_h.transpose(-2, -1)) * self.scale
        # (B * H) x num_heads x W x W
        attn_score_w = (q_w @ k_w.transpose(-2, -1)) * self.scale

        # avoid calculate self twice
        INF = -torch.diag(torch.tensor(float("inf")).repeat(W), 0) \
            .unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(B, H, self.num_heads, 1, 1) \
            .to(attn_score_w.device)

        # B_ori x H x num_heads x W x (H+W)
        if self.use_cls_token:
            # B x (1+H) x num_heads x W x (1+H)
            attn_score_h = attn_score_h.view(B, W, self.num_heads, 1+H, 1+H).permute(0, 3, 2, 1, 4)
            # B x H x num_heads x W x (1+H)
            attn_score_h = attn_score_h[:, 1:]
            # B x H x num_heads x W x (1+H+W)
            attn_score = torch.cat((attn_score_h,
                                    attn_score_w.view(B, H, self.num_heads, W, W) + INF
                                    ), -1)
        else:
            # B x H x num_heads x W x (H+W)
            attn_score = torch.cat((attn_score_h.view(B, W, self.num_heads, H, H).permute(0, 3, 2, 1, 4),
                                    attn_score_w.view(B, H, self.num_heads, W, W) + INF
                                    )
                                   , -1)
        attn = F.softmax(attn_score, dim=-1)
        attn = self.attn_drop(attn)

        attn_h, attn_w = torch.split(attn, [1+H if self.use_cls_token else H, W], -1)
        # attn_h: B_ori x W x num_heads x H x H,  B_ori x W x num_heads x H x (1+H)
        # attn_w: B_ori x H x num_heads x W x W
        attn_h = attn_h.permute(0, 3, 2, 1, 4)
        x_h = (attn_h.flatten(0, 1) @ v_h).transpose(1, 2).reshape(B, W, H, C).transpose(1, 2)
        x_w = (attn_w.flatten(0, 1) @ v_w).transpose(1, 2).reshape(B, H, W, C)

        # B_ori x H x W x C
        x = (x_h + x_w).reshape(B, H*W, C)

        if self.use_cls_token:
            # full attention for cls token
            q_cls = q_cls.view(B, self.num_heads, 1, C // self.num_heads)
            attn_score_cls = (q_cls @ k_full.transpose(-2, -1)) * self.scale
            attn_cls = F.softmax(attn_score_cls, dim=-1)
            attn_cls = self.attn_drop(attn_cls)
            x_cls = (attn_cls @ v_full).transpose(1, 2).reshape(B, 1, C)
            x = torch.cat((x_cls, x), 1)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x, prev

    @staticmethod
    def compute_macs(module, input, output):
        # T: num_token
        # S: num_token
        input = input[0]
        _, T, C = input.shape
        T_ori = T
        H = W = int(math.sqrt(T))
        macs = 0
        n_params = 0

        # Scaled-dot-product macs
        # H direction: [B x W x (1+H) x C] x [B x W x (1+H) x C] --> [B x W x (1+H) x (1+H)]
        # W direction: [B x H x W x C] x [B x H x W x C] --> [B x H x W x W]
        # cls: [B x 1 x C] x [B x T_ori x C] -> [B x 1 x T_ori]
        num_macs_kq = W * (H+1) * (H+1) * C + H * W * W * C + 1 * T_ori * C
        # [B x T x S] x [B x S x C] --> [B x T x C]
        num_macs_v = W * (1+H) * (1+H) * C + H * W * W * C + 1 * T_ori * C

        macs += num_macs_kq + num_macs_v
        # print('macs att', macs)

        # self attention: T should be equal to S
        assert W == H
        qkv_params = sum([p.numel() for p in module.qkv.parameters()])
        n_params += qkv_params
        # multiply by Seq length
        macs += qkv_params * T_ori
        # print('macs qkv', qkv_params * T_ori)

        proj_params = sum([p.numel() for p in module.proj.parameters()])
        n_params += proj_params
        macs += (proj_params * T_ori)
        # print('macs proj', proj_params * T_ori)

        module.__flops__ += macs


class Attention(nn.Module):
    def __init__(self,
                 dim,
                 in_dim=None,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 res_score=False,
                 use_avg_att=False,
                 use_avg_att_all=False,
                 use_qkvconv=False,
                 qkv_ratio=3.0,
                 add_v_res=False,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 method='conv',
                 conv_reduce_ratio=3
                 ):
        super().__init__()
        # qkv : in_dim -> dim
        if in_dim is None:
            in_dim = dim
        self.num_heads = num_heads
        self.dim = dim
        self.qkv_dim = int(dim * qkv_ratio) // 3
        head_dim = self.qkv_dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.use_conv = use_qkvconv
        self.add_v_res = add_v_res
        self.qkv_ratio = qkv_ratio

        self.downsample_ratio = 1.0

        if self.use_conv:
            self.downsample_ratio = stride
            self.method = method
            if method in ['dw_glu_pw', 'dw_bn_glu_pw', 'dw_swish_pw',
                          'dw_pw', 'dw_bn_pw']:
                # assert in_dim == dim, 'in_dim != dim not supported'
                print('in dim', in_dim, dim)
                self.dw = nn.Conv2d(in_dim, dim, kernel_size=kernel_size, padding=padding,
                                    stride=stride, bias=False, groups=in_dim)
                if 'bn' in method:
                    self.bn = nn.BatchNorm2d(dim)
                if 'swish' in method:
                    self.act = Swish()
                elif 'glu' in method:
                    self.act = nn.GELU()
                else:
                    self.act = None
                if qkv_ratio != 3:
                    # reduced dim for q,v, full dim for v
                    self.pw = nn.Conv1d(dim, self.qkv_dim * 2 + self.dim, kernel_size=1, padding=0,
                                        stride=1, bias=qkv_bias)
                else:
                    self.pw = nn.Conv1d(dim, int(dim * qkv_ratio), kernel_size=1, padding=0, stride=1, bias=qkv_bias)
            elif method == 'conv':
                self.qkv = nn.Conv2d(in_dim, int(dim * qkv_ratio), kernel_size=kernel_size, padding=padding,
                                     stride=stride, bias=qkv_bias)
            elif method in ['pw_glu_conv_glu_pw', 'pw_glu_conv_bn_glu_pw',
                            'pw_glu_conv_swish_pw',
                            'pw_glu_conv_bn_swish_pw',
                            'pw_glu_dw_glu_pw',
                            'pw_glu_conv_bn_pw',
                            ]:
                self.pw0 = nn.Conv1d(in_dim, int(dim // conv_reduce_ratio), kernel_size=1, padding=0,
                                     stride=1, bias=False)
                self.act = nn.GELU()
                if 'dw' in method:
                    self.dw = nn.Conv2d(int(dim // conv_reduce_ratio), int(dim // conv_reduce_ratio),
                                        kernel_size=kernel_size, padding=padding,
                                        stride=stride, bias=False,
                                        groups=int(dim // conv_reduce_ratio))
                else:
                    self.conv = nn.Conv2d(int(dim // conv_reduce_ratio), int(dim // conv_reduce_ratio),
                                          kernel_size=kernel_size, padding=padding,
                                          stride=stride, bias=False,
                                          groups=1)
                if 'bn' in method:
                    self.bn = nn.BatchNorm2d(int(dim // conv_reduce_ratio))
                if method == 'pw_glu_conv_bn_pw':
                    self.act2 = None
                elif 'swish' in method:
                    self.act2 = Swish()
                elif 'glu' in method:
                    self.act2 = nn.GELU()
                else:
                    self.act2 = None
                self.pw = nn.Conv1d(int(dim // conv_reduce_ratio), int(dim * qkv_ratio),
                                    kernel_size=1, padding=0, stride=1, bias=qkv_bias)
            elif method == 'conv_pw':
                self.conv = nn.Conv2d(in_dim, dim, kernel_size=kernel_size, padding=padding,
                                      stride=stride, bias=False)
                self.act = nn.GELU()
                self.pw = nn.Conv1d(dim, int(dim * qkv_ratio), kernel_size=1, padding=0, stride=1, bias=qkv_bias)
            else:
                assert False, 'conv method {} for qkv not supported'.format(method)
        else:
            self.qkv = nn.Linear(in_dim, int(dim * qkv_ratio), bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.res_score = res_score
        self.use_avg_att = use_avg_att
        self.use_avg_att_all = use_avg_att_all

    def forward_qkvconv(self, x):
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        if H*W == N:
            use_cls_token = False
        elif H*W == N-1:
            use_cls_token = True
        else:
            assert False, "num of Token {} does not match H {}, W {}".format(N, H, W)

        if use_cls_token:
            x_cls, x = torch.split(x, [1, N-1], 1)
            x_cls = x_cls.view(B, 1, C)
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        if self.method in ['dw_glu_pw', 'dw_bn_glu_pw', 'dw_swish_pw',
                           'dw_pw', 'dw_bn_pw']:
            x = self.dw(x)
            if 'bn' in self.method:
                x = self.bn(x)
            if self.act is not None:
                x = self.act(x)
            x = x.flatten(2)
            if use_cls_token:
                x = torch.cat((x_cls.transpose(1, 2), x), 2)
            x = self.pw(x)
        elif self.method in ['pw_glu_conv_glu_pw', 'pw_glu_conv_bn_glu_pw',
                             'pw_glu_conv_swish_pw',
                             'pw_glu_conv_bn_swish_pw',
                             'pw_glu_dw_glu_pw',
                             'pw_glu_conv_bn_pw',
                             ]:
            x = self.pw0(x.flatten(2))
            if self.act is not None:
                x = self.act(x)
            if 'dw' in self.method:
                x = self.dw(x.unflatten(2, (H, W)))
            else:
                x = self.conv(x.unflatten(2, (H, W)))
            if 'bn' in self.method:
                x = self.bn(x)
            if self.act2 is not None:
                x = self.act2(x)
            x = x.flatten(2)
            if use_cls_token:
                x_cls = self.pw0(x_cls.transpose(1, 2))
                x = torch.cat((x_cls, x), 2)
            x = self.pw(x)
        elif self.method == 'conv_pw':
            x = self.conv(x)
            if self.act is not None:
                x = self.act(x)
            x = x.flatten(2)
            if use_cls_token:
                x = torch.cat((x_cls.transpose(1, 2), x), 2)
            x = self.pw(x)
        elif self.method == 'conv':
            x = self.qkv(x).flatten(2)
            if use_cls_token:
                x_cls = self.qkv(x_cls.view(B, C, 1, 1).expand(-1, -1, 3, 3)).flatten(2)[:, :, 4:5]
                x = torch.cat((x_cls, x), 2)

        # x = x.view(B, 3, self.num_heads, C // self.num_heads, N).permute(1, 0, 2, 4, 3)
        if self.qkv_ratio != 3:
            q, k, v = torch.split(x.contiguous().view(B, self.num_heads,
                                                      (self.qkv_dim * 2 + self.dim)//self.num_heads,
                                                      -1).transpose(2, 3),
                            [self.qkv_dim // self.num_heads,
                             self.qkv_dim // self.num_heads,
                             self.dim // self.num_heads], dim=3)
        else:
            x = x.contiguous().view(B, 3, self.num_heads, self.qkv_dim // self.num_heads, -1).permute(1, 0, 2, 4, 3)

            q, k, v = x[0], x[1], x[2]  # make torchscript happy (cannot use tensor as tuple)

        return q, k, v

    def forward(self, x, prev=None, return_info=None):
        info = {}
        B = x.shape[0]
        C = self.dim
        if self.use_conv:
            q, k, v = self.forward_qkvconv(x)
        else:
            qkv = self.qkv(x) \
                .reshape(B, -1, 3, self.num_heads, C // self.num_heads) \
                .permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        N = q.shape[2]

        if self.use_avg_att:
            x = v
            x = x.transpose(1, 2).reshape(B, N, C)
            x[:, 0] = x[:, 1:].mean(1)
        elif self.use_avg_att_all:
            x = v
            x = x.transpose(1, 2).reshape(B, N, C)
            x = x.mean(1, keepdim=True).repeat(1, N, 1)
        else:
            attn_score = (q @ k.transpose(-2, -1)) * self.scale

            if prev is not None and self.res_score:
                attn_score = attn_score + prev

            if self.res_score:
                prev = attn_score

            attn = F.softmax(attn_score, dim=-1)

            attn = self.attn_drop(attn)
            if return_info is not None and 'attmap' in return_info:
                info['attmap'] = attn

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        if self.add_v_res:
            x = x + v.transpose(1, 2).flatten(2)

        return x, prev, info

    @staticmethod
    def compute_macs(module, input, output):
        # T: num_token
        # S: num_token
        input = input[0]
        _, T, C = input.shape
        H = W = int(math.sqrt(T))
        res = T - H*W
        H /= module.downsample_ratio
        W /= module.downsample_ratio
        T = H * W + res

        C = module.dim
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
        if module.use_conv:
            if module.method in ['dw_glu_pw', 'dw_bn_glu_pw', 'dw_swish_pw',
                                 'dw_pw',
                                 'conv_pw',
                                 'pw_glu_conv_glu_pw', 'pw_glu_conv_bn_glu_pw',
                                 'pw_glu_conv_swish_pw',
                                 'pw_glu_conv_bn_swish_pw',
                                 'pw_glu_dw_glu_pw', 'dw_bn_pw', 'pw_glu_conv_bn_pw'
                                 ]:

                H = W = int(math.sqrt(T))
                if 'conv' in module.method:
                    dw_params = sum([p.numel() for p in module.conv.parameters()])
                else:
                    dw_params = sum([p.numel() for p in module.dw.parameters()])
                dw_macs = dw_params * H * W

                if module.method[:2] == 'pw':
                    dw_params = sum([p.numel() for p in module.pw0.parameters()])
                    dw_macs += dw_params * H * W


                pw_params = sum([p.numel() for p in module.pw.parameters()])
                pw_macs = pw_params * H * W

                macs += dw_macs
                macs += pw_macs
            elif module.method == 'conv':
                qkv_params = sum([p.numel() for p in module.qkv.parameters()])
                n_params += qkv_params
                # multiply by Seq length
                macs += qkv_params * T
        else:
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

class DownAttention(nn.Module):
    def __init__(self,
                 dim,
                 in_dim=None,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 res_score=False,
                 use_avg_att=False,
                 use_avg_att_all=False,
                 use_qkvconv=False,
                 qkv_ratio=3.0,
                 add_v_res=False,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 method='conv',
                 conv_reduce_ratio=3,
                 down_resolution=False,
                 add_v_full_after_proj=False
                 ):
        super().__init__()
        # qkv : in_dim -> dim
        if in_dim is None:
            in_dim = dim
        self.num_heads = num_heads
        self.dim = dim
        self.qkv_dim = int(dim * qkv_ratio) // 3
        head_dim = self.qkv_dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.use_conv = use_qkvconv
        self.add_v_res = add_v_res
        self.qkv_ratio =qkv_ratio
        self.down_resolution = down_resolution

        self.downsample_ratio = 1.0
        self.add_v_full_after_proj = add_v_full_after_proj

        if self.use_conv:
            self.downsample_ratio = stride
            self.method = method
            if method in ['dw_glu_pw', 'dw_bn_glu_pw', 'dw_swish_pw',
                          'dw_pw', 'dw_bn_pw']:
                # assert in_dim == dim, 'in_dim != dim not supported'
                self.dw = nn.Conv2d(in_dim, dim, kernel_size=kernel_size, padding=padding,
                                    stride=stride, bias=False, groups=in_dim)
                if 'bn' in method:
                    self.bn = nn.BatchNorm2d(dim)
                if 'swish' in method:
                    self.act = Swish()
                elif 'glu' in method:
                    self.act = nn.GELU()
                else:
                    self.act = None
                if qkv_ratio != 3:
                    # reduced dim for q,v, full dim for v
                    self.pw = nn.Conv1d(dim, self.qkv_dim * 2 + self.dim, kernel_size=1, padding=0,
                                        stride=1, bias=qkv_bias)
                else:
                    self.pw = nn.Conv1d(dim, int(dim * qkv_ratio), kernel_size=1, padding=0, stride=1, bias=qkv_bias)


                self.v_dw = nn.Conv2d(in_dim, dim, kernel_size=3, padding=1, stride=1, bias=False, groups=in_dim)
                self.v_bn = nn.BatchNorm2d(dim)
                self.v_pw = nn.Conv1d(dim, dim, kernel_size=1, padding=0, stride=1, bias=qkv_bias)

            elif method == 'conv':
                self.qkv = nn.Conv2d(in_dim, int(dim * qkv_ratio), kernel_size=kernel_size, padding=padding,
                                     stride=stride, bias=qkv_bias)
            elif method in ['pw_glu_conv_glu_pw', 'pw_glu_conv_bn_glu_pw',
                            'pw_glu_conv_swish_pw',
                            'pw_glu_conv_bn_swish_pw',
                            'pw_glu_dw_glu_pw',
                            'pw_glu_conv_bn_pw',
                            ]:
                self.pw0 = nn.Conv1d(in_dim, int(dim // conv_reduce_ratio), kernel_size=1, padding=0,
                                     stride=1, bias=False)
                self.act = nn.GELU()
                if 'dw' in method:
                    self.dw = nn.Conv2d(int(dim // conv_reduce_ratio), int(dim // conv_reduce_ratio),
                                        kernel_size=kernel_size, padding=padding,
                                        stride=stride, bias=False,
                                        groups=int(dim // conv_reduce_ratio))
                else:
                    self.conv = nn.Conv2d(int(dim // conv_reduce_ratio), int(dim // conv_reduce_ratio),
                                          kernel_size=kernel_size, padding=padding,
                                          stride=stride, bias=False,
                                          groups=1)
                if 'bn' in method:
                    self.bn = nn.BatchNorm2d(int(dim // conv_reduce_ratio))
                if method == 'pw_glu_conv_bn_pw':
                    self.act2 = None
                elif 'swish' in method:
                    self.act2 = Swish()
                elif 'glu' in method:
                    self.act2 = nn.GELU()
                else:
                    self.act2 = None
                self.pw = nn.Conv1d(int(dim // conv_reduce_ratio), int(dim * qkv_ratio),
                                    kernel_size=1, padding=0, stride=1, bias=qkv_bias)
            elif method == 'conv_pw':
                self.conv = nn.Conv2d(in_dim, dim, kernel_size=kernel_size, padding=padding,
                                      stride=stride, bias=False)
                self.act = nn.GELU()
                self.pw = nn.Conv1d(dim, int(dim * qkv_ratio), kernel_size=1, padding=0, stride=1, bias=qkv_bias)
            else:
                assert False, 'conv method {} for qkv not supported'.format(method)
        else:
            self.qkv = nn.Linear(in_dim, int(dim * qkv_ratio), bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.res_score = res_score
        self.use_avg_att = use_avg_att
        self.use_avg_att_all = use_avg_att_all

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward_qkvconv(self, x):
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        if H*W == N:
            use_cls_token = False
        elif H*W == N-1:
            use_cls_token = True
        else:
            assert False, "num of Token {} does not match H {}, W {}".format(N, H, W)

        if use_cls_token:
            x_cls, x = torch.split(x, [1, N-1], 1)
            x_cls = x_cls.view(B, 1, C)
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        if self.method in ['dw_glu_pw', 'dw_bn_glu_pw', 'dw_swish_pw',
                           'dw_pw', 'dw_bn_pw']:

            v_full = self.v_dw(x)
            v_full = self.v_bn(v_full)
            v_full = self.v_pw(v_full.flatten(2))
            v_full = v_full.contiguous().transpose(1, 2)  # N x T x C

            x = self.dw(x)
            if 'bn' in self.method:
                x = self.bn(x)
            if self.act is not None:
                x = self.act(x)
            x = x.flatten(2)
            if use_cls_token:
                x = torch.cat((x_cls.transpose(1, 2), x), 2)
            x = self.pw(x)
        elif self.method in ['pw_glu_conv_glu_pw', 'pw_glu_conv_bn_glu_pw',
                             'pw_glu_conv_swish_pw',
                             'pw_glu_conv_bn_swish_pw',
                             'pw_glu_dw_glu_pw',
                             'pw_glu_conv_bn_pw',
                             ]:
            x = self.pw0(x.flatten(2))
            if self.act is not None:
                x = self.act(x)
            if 'dw' in self.method:
                x = self.dw(x.unflatten(2, (H, W)))
            else:
                x = self.conv(x.unflatten(2, (H, W)))
            if 'bn' in self.method:
                x = self.bn(x)
            if self.act2 is not None:
                x = self.act2(x)
            x = x.flatten(2)
            if use_cls_token:
                x_cls = self.pw0(x_cls.transpose(1, 2))
                x = torch.cat((x_cls, x), 2)
            x = self.pw(x)
        elif self.method == 'conv_pw':
            x = self.conv(x)
            if self.act is not None:
                x = self.act(x)
            x = x.flatten(2)
            if use_cls_token:
                x = torch.cat((x_cls.transpose(1, 2), x), 2)
            x = self.pw(x)
        elif self.method == 'conv':
            x = self.qkv(x).flatten(2)
            if use_cls_token:
                x_cls = self.qkv(x_cls.view(B, C, 1, 1).expand(-1, -1, 3, 3)).flatten(2)[:, :, 4:5]
                x = torch.cat((x_cls, x), 2)

        # x = x.view(B, 3, self.num_heads, C // self.num_heads, N).permute(1, 0, 2, 4, 3)
        if self.qkv_ratio != 3:
            q, k, v = torch.split(x.contiguous().view(B, self.num_heads,
                                                      (self.qkv_dim * 2 + self.dim)//self.num_heads,
                                                      -1).transpose(2, 3),
                                  [self.qkv_dim // self.num_heads,
                                   self.qkv_dim // self.num_heads,
                                   self.dim // self.num_heads], dim=3)
        else:
            x = x.contiguous().view(B, 3, self.num_heads, self.qkv_dim // self.num_heads, -1).permute(1, 0, 2, 4, 3)

            q, k, v = x[0], x[1], x[2]  # make torchscript happy (cannot use tensor as tuple)

        return q, k, v, v_full

    def forward(self, x, prev=None, return_info=None):
        B, N, C = x.shape
        H = W = int(math.sqrt(N))


        if H*W == N:
            use_cls_token = False
        elif H*W == N-1:
            use_cls_token = True
        else:
            assert False, "num of Token {} does not match H {}, W {}".format(N, H, W)
        info = {}
        B = x.shape[0]
        C = self.dim
        if self.use_conv:
            q, k, v, v_full = self.forward_qkvconv(x)
        else:
            qkv = self.qkv(x) \
                .reshape(B, -1, 3, self.num_heads, C // self.num_heads) \
                .permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        N = q.shape[2]

        if self.use_avg_att:
            x = v
            x = x.transpose(1, 2).reshape(B, N, C)
            x[:, 0] = x[:, 1:].mean(1)
        elif self.use_avg_att_all:
            x = v
            x = x.transpose(1, 2).reshape(B, N, C)
            x = x.mean(1, keepdim=True).repeat(1, N, 1)
        else:
            attn_score = (q @ k.transpose(-2, -1)) * self.scale

            if prev is not None and self.res_score:
                attn_score = attn_score + prev

            if self.res_score:
                prev = attn_score

            attn = F.softmax(attn_score, dim=-1)

            attn = self.attn_drop(attn)
            if return_info is not None and 'attmap' in return_info:
                info['attmap'] = attn

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        if use_cls_token:
            x_cls, x = torch.split(x, [1, N-1], 1)

        H = W = int(math.sqrt(N))
        x = x.view(B, H, W, C).contiguous().permute(0, 3, 1, 2)
        x = self.upsample(x)
        x = x.flatten(2).contiguous().transpose(1, 2)

        if not self.add_v_full_after_proj:
            x = x + v_full

        if use_cls_token:
            x = torch.cat((x_cls, x), 1)

        x = self.proj(x)
        x = self.proj_drop(x)

        if self.add_v_res:
            x = x + v.transpose(1, 2).flatten(2)

        if self.add_v_full_after_proj:
            x_cls, x = torch.split(x, [1, v_full.shape[1]], 1)
            x = x + v_full
            x = torch.cat((x_cls, x), 1)

        return x, prev, info

    @staticmethod
    def compute_macs(module, input, output):
        # T: num_token
        # S: num_token
        input = input[0]
        _, T, C = input.shape
        H = W = int(math.sqrt(T))

        upsample_macs = H*W*module.dim

        res = T - H*W
        H /= module.downsample_ratio
        W /= module.downsample_ratio
        T = H * W + res

        C = module.dim
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
        if module.use_conv:
            if module.method in ['dw_glu_pw', 'dw_bn_glu_pw', 'dw_swish_pw',
                                 'dw_pw',
                                 'conv_pw',
                                 'pw_glu_conv_glu_pw', 'pw_glu_conv_bn_glu_pw',
                                 'pw_glu_conv_swish_pw',
                                 'pw_glu_conv_bn_swish_pw',
                                 'pw_glu_dw_glu_pw', 'dw_bn_pw', 'pw_glu_conv_bn_pw'
                                 ]:

                H = W = int(math.sqrt(T))
                if 'conv' in module.method:
                    dw_params = sum([p.numel() for p in module.conv.parameters()])
                else:
                    dw_params = sum([p.numel() for p in module.dw.parameters()])
                dw_macs = dw_params * H * W

                if module.method[:2] == 'pw':
                    dw_params = sum([p.numel() for p in module.pw0.parameters()])
                    dw_macs += dw_params * H * W


                pw_params = sum([p.numel() for p in module.pw.parameters()])
                pw_macs = pw_params * H * W

                # for v_full
                dw_params = sum([p.numel() for p in module.v_dw.parameters()])
                dw_macs += dw_params * H * W

                pw_params = sum([p.numel() for p in module.v_pw.parameters()])
                pw_macs += pw_params * H * W

                macs += dw_macs
                macs += pw_macs

            elif module.method == 'conv':
                qkv_params = sum([p.numel() for p in module.qkv.parameters()])
                n_params += qkv_params
                # multiply by Seq length
                macs += qkv_params * T
        else:
            qkv_params = sum([p.numel() for p in module.qkv.parameters()])
            n_params += qkv_params
            # multiply by Seq length
            macs += qkv_params * T
            # print('macs qkv', qkv_params * T / 1e8)

        proj_params = sum([p.numel() for p in module.proj.parameters()])
        n_params += proj_params
        macs += (proj_params * T)
        # print('macs proj', proj_params * T / 1e8)

        macs += upsample_macs

        module.__flops__ += macs
        # return n_params, macs

class CrossScaleAttention(nn.Module):
    def __init__(self,
                 dim,
                 in_dim=None,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 res_score=False,
                 use_avg_att=False,
                 use_avg_att_all=False,
                 use_qkvconv=False,
                 qkv_ratio=3.0,
                 add_v_res=False,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 method='conv',
                 conv_reduce_ratio=3,
                 down_resolution=False,
                 add_v_full_after_proj=False
                 ):
        super().__init__()
        # qkv : in_dim -> dim
        if in_dim is None:
            in_dim = dim
        self.num_heads = num_heads
        self.dim = dim
        self.qkv_dim = int(dim * qkv_ratio) // 3
        head_dim = self.qkv_dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.use_conv = use_qkvconv
        self.add_v_res = add_v_res
        self.qkv_ratio =qkv_ratio
        self.down_resolution = down_resolution

        self.downsample_ratio = 1.0
        self.add_v_full_after_proj = add_v_full_after_proj

        if self.use_conv:
            self.downsample_ratio = stride
            self.method = method
            if method in ['dw_glu_pw', 'dw_bn_glu_pw', 'dw_swish_pw',
                          'dw_pw', 'dw_bn_pw']:
                # assert in_dim == dim, 'in_dim != dim not supported'
                self.kv_dw = nn.Conv2d(in_dim, dim, kernel_size=kernel_size, padding=padding,
                                    stride=stride, bias=False, groups=in_dim)
                if 'bn' in method:
                    self.kv_bn = nn.BatchNorm2d(dim)
                if 'swish' in method:
                    self.act = Swish()
                elif 'glu' in method:
                    self.act = nn.GELU()
                else:
                    self.act = None
                assert qkv_ratio == 3
                self.kv_pw = nn.Conv1d(dim, int(dim * 2), kernel_size=1, padding=0, stride=1, bias=qkv_bias)

                # q
                self.q_dw = nn.Conv2d(in_dim, dim, kernel_size=kernel_size, padding=padding,
                                       stride=1, bias=False, groups=in_dim)
                self.q_bn = nn.BatchNorm2d(dim)
                self.q_pw = nn.Conv1d(dim, dim, kernel_size=1, padding=0, stride=1, bias=qkv_bias)

                # v_full
                self.v_dw = nn.Conv2d(in_dim, dim, kernel_size=3, padding=1, stride=1, bias=False, groups=in_dim)
                self.v_bn = nn.BatchNorm2d(dim)
                self.v_pw = nn.Conv1d(dim, dim, kernel_size=1, padding=0, stride=1, bias=qkv_bias)

            elif method == 'conv':
                self.qkv = nn.Conv2d(in_dim, int(dim * qkv_ratio), kernel_size=kernel_size, padding=padding,
                                     stride=stride, bias=qkv_bias)
            elif method in ['pw_glu_conv_glu_pw', 'pw_glu_conv_bn_glu_pw',
                            'pw_glu_conv_swish_pw',
                            'pw_glu_conv_bn_swish_pw',
                            'pw_glu_dw_glu_pw',
                            'pw_glu_conv_bn_pw',
                            ]:
                self.pw0 = nn.Conv1d(in_dim, int(dim // conv_reduce_ratio), kernel_size=1, padding=0,
                                     stride=1, bias=False)
                self.act = nn.GELU()
                if 'dw' in method:
                    self.dw = nn.Conv2d(int(dim // conv_reduce_ratio), int(dim // conv_reduce_ratio),
                                        kernel_size=kernel_size, padding=padding,
                                        stride=stride, bias=False,
                                        groups=int(dim // conv_reduce_ratio))
                else:
                    self.conv = nn.Conv2d(int(dim // conv_reduce_ratio), int(dim // conv_reduce_ratio),
                                          kernel_size=kernel_size, padding=padding,
                                          stride=stride, bias=False,
                                          groups=1)
                if 'bn' in method:
                    self.bn = nn.BatchNorm2d(int(dim // conv_reduce_ratio))
                if method == 'pw_glu_conv_bn_pw':
                    self.act2 = None
                elif 'swish' in method:
                    self.act2 = Swish()
                elif 'glu' in method:
                    self.act2 = nn.GELU()
                else:
                    self.act2 = None
                self.pw = nn.Conv1d(int(dim // conv_reduce_ratio), int(dim * qkv_ratio),
                                    kernel_size=1, padding=0, stride=1, bias=qkv_bias)
            elif method == 'conv_pw':
                self.conv = nn.Conv2d(in_dim, dim, kernel_size=kernel_size, padding=padding,
                                      stride=stride, bias=False)
                self.act = nn.GELU()
                self.pw = nn.Conv1d(dim, int(dim * qkv_ratio), kernel_size=1, padding=0, stride=1, bias=qkv_bias)
            else:
                assert False, 'conv method {} for qkv not supported'.format(method)
        else:
            self.qkv = nn.Linear(in_dim, int(dim * qkv_ratio), bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.res_score = res_score
        self.use_avg_att = use_avg_att
        self.use_avg_att_all = use_avg_att_all

        # self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward_qkvconv(self, x):
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        if H*W == N:
            use_cls_token = False
        elif H*W == N-1:
            use_cls_token = True
        else:
            assert False, "num of Token {} does not match H {}, W {}".format(N, H, W)

        if use_cls_token:
            x_cls, x = torch.split(x, [1, N-1], 1)
            x_cls = x_cls.view(B, 1, C)
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        if self.method in ['dw_glu_pw', 'dw_bn_glu_pw', 'dw_swish_pw',
                           'dw_pw', 'dw_bn_pw']:

            v_full = self.v_dw(x)
            v_full = self.v_bn(v_full)
            v_full = self.v_pw(v_full.flatten(2))
            # v_full = v_full.contiguous().transpose(1, 2)  # N x T x C

            q = self.q_dw(x)
            q = self.q_bn(q).flatten(2)
            if use_cls_token:
                q = torch.cat((x_cls.transpose(1, 2), q), 2)
            q = self.q_pw(q)
            # q = q.contiguous().transpose(1, 2)  # N x T x C

            kv = self.kv_dw(x)
            if 'bn' in self.method:
                kv = self.kv_bn(kv)
            if self.act is not None:
                kv = self.act(kv)
            kv = kv.flatten(2)
            if use_cls_token:
                kv = torch.cat((x_cls.transpose(1, 2), kv), 2)
            kv = self.kv_pw(kv)
            # kv = kv.contiguous().transpose(1, 2)

        elif self.method in ['pw_glu_conv_glu_pw', 'pw_glu_conv_bn_glu_pw',
                             'pw_glu_conv_swish_pw',
                             'pw_glu_conv_bn_swish_pw',
                             'pw_glu_dw_glu_pw',
                             'pw_glu_conv_bn_pw',
                             ]:
            x = self.pw0(x.flatten(2))
            if self.act is not None:
                x = self.act(x)
            if 'dw' in self.method:
                x = self.dw(x.unflatten(2, (H, W)))
            else:
                x = self.conv(x.unflatten(2, (H, W)))
            if 'bn' in self.method:
                x = self.bn(x)
            if self.act2 is not None:
                x = self.act2(x)
            x = x.flatten(2)
            if use_cls_token:
                x_cls = self.pw0(x_cls.transpose(1, 2))
                x = torch.cat((x_cls, x), 2)
            x = self.pw(x)
        elif self.method == 'conv_pw':
            x = self.conv(x)
            if self.act is not None:
                x = self.act(x)
            x = x.flatten(2)
            if use_cls_token:
                x = torch.cat((x_cls.transpose(1, 2), x), 2)
            x = self.pw(x)
        elif self.method == 'conv':
            x = self.qkv(x).flatten(2)
            if use_cls_token:
                x_cls = self.qkv(x_cls.view(B, C, 1, 1).expand(-1, -1, 3, 3)).flatten(2)[:, :, 4:5]
                x = torch.cat((x_cls, x), 2)


        # x = x.contiguous().view(B, 3, self.num_heads, self.qkv_dim // self.num_heads, -1).permute(1, 0, 2, 4, 3)
        #
        # q, k, v = x[0], x[1], x[2]  # make torchscript happy (cannot use tensor as tuple)

        k, v = torch.chunk(kv, 2, dim=1)

        q = q.view(B, -1, self.num_heads, self.qkv_dim // self.num_heads).transpose(1, 2)
        k = k.view(B, -1, self.num_heads, self.qkv_dim // self.num_heads).transpose(1, 2)
        v = v.view(B, -1, self.num_heads, self.qkv_dim // self.num_heads).transpose(1, 2)
        v_full = v_full.transpose(1, 2)

        return q, k, v, v_full

    def forward(self, x, prev=None, return_info=None):
        B, N, C = x.shape
        H = W = int(math.sqrt(N))


        if H*W == N:
            use_cls_token = False
        elif H*W == N-1:
            use_cls_token = True
        else:
            assert False, "num of Token {} does not match H {}, W {}".format(N, H, W)
        info = {}
        B = x.shape[0]
        C = self.dim
        if self.use_conv:
            q, k, v, v_full = self.forward_qkvconv(x)
        else:
            qkv = self.qkv(x) \
                .reshape(B, -1, 3, self.num_heads, C // self.num_heads) \
                .permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        N = q.shape[2]

        if self.use_avg_att:
            x = v
            x = x.transpose(1, 2).reshape(B, N, C)
            x[:, 0] = x[:, 1:].mean(1)
        elif self.use_avg_att_all:
            x = v
            x = x.transpose(1, 2).reshape(B, N, C)
            x = x.mean(1, keepdim=True).repeat(1, N, 1)
        else:
            attn_score = (q @ k.transpose(-2, -1)) * self.scale

            if prev is not None and self.res_score:
                attn_score = attn_score + prev

            if self.res_score:
                prev = attn_score

            attn = F.softmax(attn_score, dim=-1)

            attn = self.attn_drop(attn)
            if return_info is not None and 'attmap' in return_info:
                info['attmap'] = attn

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        if use_cls_token:
            x_cls, x = torch.split(x, [1, N-1], 1)

        # H = W = int(math.sqrt(N))
        # x = x.view(B, H, W, C).contiguous().permute(0, 3, 1, 2)
        # x = self.upsample(x)
        # x = x.flatten(2).contiguous().transpose(1, 2)

        if not self.add_v_full_after_proj:
            x = x + v_full

        if use_cls_token:
            x = torch.cat((x_cls, x), 1)

        x = self.proj(x)
        x = self.proj_drop(x)

        if self.add_v_res:
            x = x + v.transpose(1, 2).flatten(2)

        if self.add_v_full_after_proj:
            x_cls, x = torch.split(x, [1, v_full.shape[1]], 1)
            x = x + v_full
            x = torch.cat((x_cls, x), 1)

        return x, prev, info

    @staticmethod
    def compute_macs(module, input, output):
        # T: num_token
        # S: num_token
        input = input[0]
        _, T, C = input.shape
        T_ori = T
        H = W = int(math.sqrt(T))


        res = T - H*W
        H /= module.downsample_ratio
        W /= module.downsample_ratio
        T = H * W + res

        C = module.dim
        S = T
        macs = 0
        n_params = 0

        # Scaled-dot-product macs
        # [B x T x C] x [B x C x S] --> [B x T x S]
        # multiplication-addition is counted as 1 because operations can be fused
        num_macs_kq = T_ori * T * C
        # [B x T x S] x [B x S x C] --> [B x T x C]
        num_macs_v = T_ori * C * T

        macs += num_macs_kq + num_macs_v
        # print('macs att', macs / 1e8)

        # self attention: T should be equal to S
        assert T == S
        if module.use_conv:
            if module.method in ['dw_glu_pw', 'dw_bn_glu_pw', 'dw_swish_pw',
                                 'dw_pw',
                                 'conv_pw',
                                 'pw_glu_conv_glu_pw', 'pw_glu_conv_bn_glu_pw',
                                 'pw_glu_conv_swish_pw',
                                 'pw_glu_conv_bn_swish_pw',
                                 'pw_glu_dw_glu_pw', 'dw_bn_pw', 'pw_glu_conv_bn_pw'
                                 ]:

                H = W = int(math.sqrt(T))

                dw_params = sum([p.numel() for p in module.kv_dw.parameters()])
                dw_macs = dw_params * H * W

                dw_params = sum([p.numel() for p in module.q_dw.parameters()])
                dw_macs += dw_params * T_ori

                dw_params = sum([p.numel() for p in module.v_dw.parameters()])
                dw_macs += dw_params * (T_ori - 1)


                pw_params = sum([p.numel() for p in module.kv_pw.parameters()])
                pw_macs = pw_params * H * W

                pw_params = sum([p.numel() for p in module.q_pw.parameters()])
                pw_macs += pw_params * T_ori

                pw_params = sum([p.numel() for p in module.v_pw.parameters()])
                pw_macs += pw_params * (T_ori - 1)

                macs += dw_macs
                macs += pw_macs

            elif module.method == 'conv':
                qkv_params = sum([p.numel() for p in module.qkv.parameters()])
                n_params += qkv_params
                # multiply by Seq length
                macs += qkv_params * T
        else:
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
