import math
from functools import partial
import logging
import torch
from torch import nn
from timm.models.layers import DropPath, trunc_normal_, to_2tuple
from layers import (
    Long2DSCSelfAttention,
    FastAttention,
    PerformerSelfAttention,
    LinformerSelfAttention,
    SRSelfAttention
)
from layers.se_layer import SELayer_Seq, SELayer_ECA
from .registry import register_model
from math import sqrt

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
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
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0.,
                 rpe=False, wx=14, wy=14, nglo=1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Inspired by swin transformer:
        # https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py#L88-L103
        # define parameter tables for local and global relative position bias
        self.rpe = rpe
        if rpe:
            self.wx = wx
            self.wy = wy
            self.nglo = nglo
            self.local_relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * wx - 1) * (2 * wy - 1),
                            num_heads))  # (2*wx-1, 2*wy-1, nH)
            trunc_normal_(self.local_relative_position_bias_table, std=.02)
            if nglo >= 1:
                self.g2l_relative_position_bias = nn.Parameter(
                    torch.zeros(2, num_heads, nglo))  # (2, nH, nglo)
                self.g2g_relative_position_bias = nn.Parameter(
                    torch.zeros(num_heads, nglo, nglo))  # (nH, nglo, nglo)
                trunc_normal_(self.g2l_relative_position_bias, std=.02)
                trunc_normal_(self.g2g_relative_position_bias, std=.02)

            # get pair-wise relative position index
            coords_h = torch.arange(wx)
            coords_w = torch.arange(wy)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, wx, wy
            coords_flatten = torch.flatten(coords, 1)  # 2, Wx*Wy
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wx*Wy, Wx*Wy
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wx*Wy, Wx*Wy, 2
            relative_coords[:, :, 0] += wx - 1  # shift to start from 0
            relative_coords[:, :, 1] += wy - 1
            relative_coords[:, :, 0] *= 2 * wy - 1
            relative_position_index = relative_coords.sum(-1)  # Wx*Wy, Wx*Wy
            self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x, nx=None, ny=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if self.rpe:
            wx = wy = int(sqrt(N - self.nglo)) # current feature map size
            local_relative_position_bias = self.local_relative_position_bias_table[
                self.relative_position_index.view(-1)].view(
                self.wx*self.wy, self.wx*self.wy, -1)  # Wh*Ww, Wh*Ww,nH

            # print(f'before: x.shape {x.shape} self.nglo {self.nglo} wx {wx} wy {wy} + self.wx { self.wx} self.wy  {self.wy} + local_relative_position_bias {local_relative_position_bias.shape}')

            local_relative_position_bias = self.interpolate_pos_encoding(x, local_relative_position_bias)

            # print(f'after: x.shape {x.shape} self.nglo {self.nglo} wx {wx} wy {wy} + self.wx { self.wx} self.wy  {self.wy} + local_relative_position_bias {local_relative_position_bias.shape}')

            # assert N == self.nglo + self.wx*self.wy, "For relative position, N != self.nglo + self.wx*self.wy!"

            relative_position_bias = local_relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

            if self.nglo > 0:
                # relative position embedding of global tokens
                global_relative_position_bias = torch.cat([
                    self.g2g_relative_position_bias,
                    self.g2l_relative_position_bias[0].unsqueeze(-1).expand(-1, -1, wx*wy)
                ], dim=-1)  # nH, nglo, N
                # relative position embedding of local tokens
                local_relative_position_bias = torch.cat([
                    self.g2l_relative_position_bias[1].unsqueeze(1).expand(-1, wx*wy, -1),
                    relative_position_bias,
                ], dim=-1)  # nH, Wh*Ww, N
                relative_position_bias = torch.cat([
                    global_relative_position_bias,
                    local_relative_position_bias,
                ], dim=1)  # nH, N, N
            attn = attn + relative_position_bias.unsqueeze(0)

        attn = (attn - torch.max(attn, dim=-1, keepdim=True)[0]).softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


    def interpolate_pos_encoding(self, x, pos_embed):
        B, N, C = x.shape
        npatch = N - self.nglo
        wx = wy = int(sqrt(npatch)) # current feature map size

        num_xy_w, num_xy_h, nhead = pos_embed.shape # existing position embedding size

        if npatch == num_xy_w:
            return pos_embed

        pos_embed = nn.functional.interpolate(
            pos_embed.reshape(1, num_xy_w, num_xy_h, nhead).permute(0, 3, 1, 2).contiguous(),
            scale_factor=npatch / num_xy_w,
            mode='bicubic',
        )

        pos_embed = pos_embed.permute(0, 2, 3, 1).contiguous().squeeze(0)
        return pos_embed


    @staticmethod
    def compute_macs(module, input, output):
        # T: num_token
        # S: num_token
        input = input[0]
        _, T, C = input.shape
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


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, patch_size, nx, ny, in_chans=3, embed_dim=768, nglo=1,
                 norm_layer=nn.LayerNorm, norm_embed=True, drop_rate=0.0,
                 ape=True):
        # maximal global/x-direction/y-direction tokens: nglo, nx, ny
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=patch_size)

        self.norm_embed = norm_layer(embed_dim) if norm_embed else None

        self.nx = nx
        self.ny = ny
        self.Nglo = nglo
        if nglo >= 1:
            self.cls_token = nn.Parameter(torch.zeros(1, nglo, embed_dim))
            trunc_normal_(self.cls_token, std=.02)
        else:
            self.cls_token = None
        self.ape = ape
        if ape:
            self.cls_pos_embed = nn.Parameter(torch.zeros(1, nglo, embed_dim))
            self.x_pos_embed = nn.Parameter(torch.zeros(1, nx, embed_dim // 2))
            self.y_pos_embed = nn.Parameter(torch.zeros(1, ny, embed_dim // 2))
            trunc_normal_(self.cls_pos_embed, std=.02)
            trunc_normal_(self.x_pos_embed, std=.02)
            trunc_normal_(self.y_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

    def forward(self, xtuple):
        x, nx, ny = xtuple
        B = x.shape[0]

        x = self.proj(x)
        nx, ny = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)
        # assert nx == self.nx and ny == self.ny, "Fix input size!"

        if self.norm_embed:
            x = self.norm_embed(x)

        # concat cls_token
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(
                B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x), dim=1)

        if self.ape:
            # add position embedding
            pos_embed_2d = torch.cat([
                self.x_pos_embed.unsqueeze(2).expand(-1, -1, self.ny, -1),
                self.y_pos_embed.unsqueeze(1).expand(-1, self.nx, -1, -1),
            ], dim=-1).flatten(start_dim=1, end_dim=2)

            pos_embed_2d = self.interpolate_pos_encoding(x, pos_embed_2d)

            x = x + torch.cat([self.cls_pos_embed, pos_embed_2d], dim=1).expand(
                B, -1, -1)

        x = self.pos_drop(x)

        return x, nx, ny

    def interpolate_pos_encoding(self, x, pos_embed_2d):
        npatch = x.shape[1]
        N = pos_embed_2d.shape[1] 
        if npatch == N:
            return pos_embed_2d

        dim = x.shape[-1]
        pos_embed_2d = nn.functional.interpolate(
            pos_embed_2d.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=math.sqrt(npatch / N),
            mode='bicubic',
        )
        pos_embed_2d = pos_embed_2d.permute(0, 2, 3, 1).contiguous().view(1, -1, dim)
        return pos_embed_2d


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# for Performer, start
def get_module_device(module):
    return next(module.parameters()).device


def find_modules(nn_module, type):
    return [module for module in nn_module.modules() if isinstance(module, type)]

# for Performer, end


class AttnBlock(nn.Module):
    """ Meta Attn Block
    """

    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None, drop=0.,
                 attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm,
                 attn_type='full', w=7, d=1, sharew=False, nglo=1,
                 only_glo=False,
                 seq_len=None, num_feats=256, share_kv=False, sw_exact=0,
                 rratio=2, rpe=False, wx=14, wy=14,
                 add_pooled=False, pool_size=1, mode=0, pool_method=None, with_se=False, se_mlp_ratio=0.625):
        super().__init__()
        self.norm = norm_layer(dim)
        if attn_type == 'full':
            self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                  qk_scale=qk_scale, attn_drop=attn_drop,
                                  proj_drop=drop,
                                  rpe=rpe, wx=wx, wy=wy, nglo=nglo)
        # elif attn_type == 'longformer_cuda':
        #     self.attn = Longformer2DSelfAttention(
        #         dim, num_heads=num_heads, qkv_bias=qkv_bias,
        #         qk_scale=qk_scale, attn_drop=attn_drop,
        #         proj_drop=drop, w=w, d=d, sharew=sharew,
        #         nglo=nglo, only_glo=only_glo)
        elif attn_type == 'longformerhand':
            self.attn = Long2DSCSelfAttention(
                dim, exact=sw_exact, num_heads=num_heads, qkv_bias=qkv_bias,
                qk_scale=qk_scale, attn_drop=attn_drop,
                proj_drop=drop, w=w, d=d, sharew=sharew,
                nglo=nglo, only_glo=only_glo, autograd=False,
                rpe=rpe, add_pooled=add_pooled, pool_size=pool_size, mode=mode, pool_method=pool_method, wx=wx, wy=wy
            )
        elif attn_type == 'longformerauto':
            self.attn = Long2DSCSelfAttention(
                dim, exact=sw_exact, num_heads=num_heads, qkv_bias=qkv_bias,
                qk_scale=qk_scale, attn_drop=attn_drop,
                proj_drop=drop, w=w, d=d, sharew=sharew,
                nglo=nglo, only_glo=only_glo, autograd=True,
                rpe=rpe, add_pooled=add_pooled, pool_size=pool_size, mode=mode
            )
        elif attn_type == 'linformer':
            assert seq_len is not None, "seq_len must be provided for Linformer!"
            self.attn = LinformerSelfAttention(
                dim, seq_len, num_feats=num_feats,
                num_heads=num_heads, qkv_bias=qkv_bias,
                qk_scale=qk_scale, attn_drop=attn_drop,
                proj_drop=drop, share_kv=share_kv,
            )
        elif attn_type == 'srformer':
            self.attn = SRSelfAttention(
                dim, rratio=rratio,
                num_heads=num_heads, qkv_bias=qkv_bias,
                qk_scale=qk_scale, attn_drop=attn_drop,
                proj_drop=drop
            )
        elif attn_type == 'performer':
            self.attn = PerformerSelfAttention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias,
                qk_scale=qk_scale, attn_drop=attn_drop,
                proj_drop=drop, nb_features=num_feats,
            )
        else:
            raise ValueError(
                "Not supported attention type {}".format(attn_type))
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        
        self.se = None
        if with_se == 'SE':
            self.se = SELayer_Seq(dim, se_mlp_ratio)
        elif with_se == 'ECA':
            k_size = int(se_mlp_ratio * dim)
            k_size = k_size + 1 if k_size%2==0 else k_size
            self.se = SELayer_ECA(dim, k_size)

    def forward(self, xtuple):
        x, nx, ny = xtuple
        out = self.attn(self.norm(x), nx, ny)
        if self.se:
            out = self.se(out.permute(0,2,1)).permute(0,2,1).contiguous()

        x = x + self.drop_path(out)
        return x, nx, ny


class MlpBlock(nn.Module):
    """ Meta MLP Block
    """

    def __init__(self, dim, out_dim=None, mlp_ratio=4., drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, balanced_mlp_ratio=0.0):
        super().__init__()
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm = norm_layer(dim)
        mlp_hidden_dim = int(dim * (mlp_ratio-balanced_mlp_ratio))
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       out_features=out_dim, act_layer=act_layer, drop=drop)
        self.shortcut = nn.Identity()
        if out_dim is not None and out_dim != dim:
            self.shortcut = nn.Sequential(nn.Linear(dim, out_dim),
                                          nn.Dropout(drop))

    def forward(self, xtuple):
        x, nx, ny = xtuple
        x = self.shortcut(x) + self.drop_path(self.mlp(self.norm(x)))
        return x, nx, ny


class MsViT(nn.Module):
    """ Multiscale Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, arch, img_size=512, in_chans=3,
                 num_classes=1000,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 norm_embed=False, w=7, d=1, sharew=False, only_glo=False,
                 share_kv=False,
                 attn_type='longformerhand', sw_exact=0, mode=0, pool_method=None, use_dense_prediction=False, with_se=False, se_mlp_ratio=0.625, se_mlp_balance=False, **args):
        super().__init__()
        self.num_classes = num_classes
        if 'ln_eps' in args:
            ln_eps = args['ln_eps']
            self.norm_layer = partial(nn.LayerNorm, eps=ln_eps)
            logging.info("Customized LayerNorm EPS: {}".format(ln_eps))
        else:
            self.norm_layer = norm_layer
        self.drop_path_rate = drop_path_rate
        self.attn_type = attn_type

        # for performer, start
        if attn_type == "performer":
            self.auto_check_redraw = True  # TODO: make this an choice
            self.feature_redraw_interval = 1
            self.register_buffer('calls_since_last_redraw', torch.tensor(0))
        # for performer, end

        self.attn_args = dict({
            'attn_type': attn_type,
            'qkv_bias': qkv_bias,
            'qk_scale': qk_scale,
            'drop': drop_rate,
            'attn_drop': attn_drop_rate,
            'w': w,
            'd': d,
            'sharew': sharew,
            'only_glo': only_glo,
            'share_kv': share_kv,
            'sw_exact': sw_exact,
            'norm_layer': norm_layer,
            'mode': mode,
            'pool_method': pool_method,
            'with_se': with_se,
            'se_mlp_ratio': se_mlp_ratio,
        })
        self.patch_embed_args = dict({
            'norm_layer': norm_layer,
            'norm_embed': norm_embed,
            'drop_rate': drop_rate,
        })

        mlp_balance_on = False
        if se_mlp_balance and with_se:
            mlp_balance_on = True

        self.mlp_args = dict({
            'mlp_ratio': 4.0,
            'norm_layer': norm_layer,
            'act_layer': nn.GELU,
            'drop': drop_rate,
            'balanced_mlp_ratio': se_mlp_ratio if mlp_balance_on else 0.0,
        })

        self.Nx = img_size
        self.Ny = img_size

        def parse_arch(arch):
            layer_cfgs = []
            for layer in arch.split('_'):
                layer_cfg = {'l': 1, 'h': 3, 'd': 192, 'n': 1, 's': 1, 'g': 1,
                             'p': 2, 'f': 7, 'a': 1, 'r': 0}  # defaults
                for attr in layer.split(','):
                    layer_cfg[attr[0]] = int(attr[1:])
                layer_cfgs.append(layer_cfg)
            return layer_cfgs

        self.layer_cfgs = parse_arch(arch)
        self.num_layers = len(self.layer_cfgs)
        self.depth = sum([cfg['n'] for cfg in self.layer_cfgs])
        self.out_planes = self.layer_cfgs[-1]['d']
        self.Nglos = [cfg['g'] for cfg in self.layer_cfgs]
        self.avg_pool = args['avg_pool'] if 'avg_pool' in args else False

        dprs = torch.linspace(0, drop_path_rate, self.depth).split(
            [cfg['n'] for cfg in self.layer_cfgs]
        )  # stochastic depth decay rule
        self.layer1 = self._make_layer(in_chans, self.layer_cfgs[0],
                                       dprs=dprs[0], layerid=1)
        self.layer2 = self._make_layer(self.layer_cfgs[0]['d'],
                                       self.layer_cfgs[1], dprs=dprs[1],
                                       layerid=2)
        self.layer3 = self._make_layer(self.layer_cfgs[1]['d'],
                                       self.layer_cfgs[2], dprs=dprs[2],
                                       layerid=3)
        if self.num_layers == 3:
            self.layer4 = None
        elif self.num_layers == 4:
            self.layer4 = self._make_layer(self.layer_cfgs[2]['d'],
                                           self.layer_cfgs[3], dprs=dprs[3],
                                           layerid=4)
        else:
            raise ValueError("Numer of layers {} not implemented yet!".format(self.num_layers))
        self.norm = norm_layer(self.out_planes)

        # Classifier head
        self.head = nn.Linear(self.out_planes,
                              num_classes) if num_classes > 0 else nn.Identity()

        # Region prediction head
        self.use_dense_prediction = use_dense_prediction
        if self.use_dense_prediction: self.head_dense = None


        self.apply(self._init_weights)

    def _make_layer(self, in_dim, layer_cfg, dprs, layerid=0):
        layer_id, num_heads, dim, num_block, is_sparse_attn, nglo, patch_size, num_feats, ape, add_pooled \
            = layer_cfg['l'], layer_cfg['h'], layer_cfg['d'], layer_cfg['n'], layer_cfg['s'], layer_cfg['g'], layer_cfg['p'], layer_cfg['f'], layer_cfg['a'], layer_cfg['r']

        assert layerid == layer_id, "Error in _make_layer: layerid {} does not equal to layer_id {}".format(layerid, layer_id)
        self.Nx = nx = self.Nx // patch_size
        self.Ny = ny = self.Ny // patch_size
        seq_len = nx * ny + nglo

        self.attn_args['nglo'] = nglo
        self.patch_embed_args['nglo'] = nglo
        self.attn_args['num_feats'] = num_feats  # shared for linformer and performer
        self.attn_args['rratio'] = num_feats  # srformer reuses this parameter
        self.attn_args['w'] = num_feats  # longformer reuses this parameter
        pool_size = self.attn_args['w'] # nx // (8 * self.attn_args['w'])

        if add_pooled and pool_size >= 1:
            logging.info("Layer {}: Infered feature pool size: {}".format(layerid, pool_size))
        if is_sparse_attn == 0:
            self.attn_args['attn_type'] = 'full'

        

        # patch embedding
        layers = [
            PatchEmbed(patch_size, nx, ny, in_chans=in_dim, embed_dim=dim, ape=ape,
                       **self.patch_embed_args)
        ]
        for dpr in dprs:
            layers.append(AttnBlock(
                dim, num_heads, drop_path=dpr, seq_len=seq_len, rpe=not ape,
                wx=nx, wy=ny,
                add_pooled=add_pooled, pool_size=pool_size,
                **self.attn_args
            ))
            layers.append(MlpBlock(dim, drop_path=dpr, **self.mlp_args))
        return nn.Sequential(*layers)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        no_decay = {'pos_embed', 'cls_token',
                    'norm.weight', 'norm.bias',
                    'norm_embed', 'head.bias',
                    'relative_position'}
        return no_decay

    def get_classifier(self):
        return self.head

    def forward_features(self, x):
        B = x.shape[0]
        x, nx, ny = self.layer1((x, None, None))
        x = x[:, self.Nglos[0]:].transpose(-2, -1).reshape(B, -1, nx, ny)

        x, nx, ny = self.layer2((x, nx, ny))
        x = x[:, self.Nglos[1]:].transpose(-2, -1).reshape(B, -1, nx, ny)

        x, nx, ny = self.layer3((x, nx, ny))

        if self.layer4 is not None:
            x = x[:, self.Nglos[2]:].transpose(-2, -1).reshape(B, -1, nx, ny)
            x, nx, ny = self.layer4((x, nx, ny))

        x = self.norm(x)  # B L C

        if self.Nglos[-1] > 0 and (not self.avg_pool):
            x_cls, x_region = x[:, 0], x[:, 1:]
        else:
            x_cls, x_region = torch.mean(x, dim=1), x

        if self.use_dense_prediction:
            return x_cls, x_region
        else:
            return x_cls


    def stage_forward_with_features(self, x_tuple, i):
        layer_dict = {1: self.layer1, 2: self.layer2, 3: self.layer3, 4: self.layer4}

        # print(f'stage {i}')
        if i == 4:
            x, nx, ny = x_tuple 
            B = x.shape[0]  
            x = x[:, self.Nglos[i-2]:].transpose(-2, -1).reshape(B, -1, nx, ny)
            x_tuple = (x, nx, ny) 

        j = 0
        output_fea = []
        for b, blk in enumerate(layer_dict[i]):
            if b > 0: j+=1
            x_tuple = blk(x_tuple)

            if j > 0 and j % 2 == 0:
                x, nx, ny = x_tuple   
                output_fea.append(x)  

        if i == 1 or i == 2:
            x, nx, ny = x_tuple   
            B = x.shape[0]
            x = x[:, self.Nglos[i-1]:].transpose(-2, -1).reshape(B, -1, nx, ny)
            x_tuple = (x, nx, ny)

        return x_tuple, output_fea

    def forward_return_n_last_blocks(self, x, n=1, return_patch_avgpool=False, depth=[]):

        num_blks = sum(depth)
        start_idx = num_blks - n

        sum_cur = 0
        for i, d in enumerate(depth):
            sum_cur_new = sum_cur + d
            if start_idx >= sum_cur and start_idx < sum_cur_new:
                start_stage = i
                start_blk = start_idx - sum_cur
            sum_cur = sum_cur_new


        # we will return the averaged token features from the `n` last blocks
        # note: there is no [CLS] token in Swin Transformer
        output = []
        s = 0
        x_tuple = (x, None, None)

        for i in range(self.num_layers):
            x_tuple, fea = self.stage_forward_with_features(x_tuple, i+1)

            if i >= start_stage:
                for x_ in fea[start_blk:]:

                    if i == self.num_layers-1: # use the norm in the last stage
                        x_ = self.norm(x_)

                    if self.Nglos[i] > 0 and (not self.avg_pool):
                        x_ =  x_[:, 0]
                    else:
                        x_ = torch.mean(x_, dim=1)                    

                    x_avg = torch.flatten(x_, 1)  # B C     
                    # print(f'Stage {i},  x_avg {x_avg.shape}')          
                    output.append(x_avg)

                start_blk = 0

        return torch.cat(output, dim=-1)


        if self.Nglos[-1] > 0 and (not self.avg_pool):
            return x[:, 0]
        else:
            return torch.mean(x, dim=1)


    def check_redraw_projections(self):
        if not self.training:
            return

        if self.calls_since_last_redraw >= self.feature_redraw_interval:
            device = get_module_device(self)
            fast_attentions = find_modules(self, FastAttention)
            for fast_attention in fast_attentions:
                fast_attention.redraw_projection_matrix(device)

            self.calls_since_last_redraw.zero_()
            return

        self.calls_since_last_redraw += 1

    def reset_vil_mode(self, mode):
        longformer_attentions = find_modules(self, Long2DSCSelfAttention)
        for longformer_attention in longformer_attentions:
            mode_old = longformer_attention.mode
            if mode_old != mode:
                longformer_attention.mode = mode
                logging.info(
                    "Change vil attention mode from {} to {} in " "layer {}"
                        .format(mode_old, mode, longformer_attention))
        return

    # def forward(self, x):
    #     if self.attn_type == "performer" and self.auto_check_redraw:
    #         self.check_redraw_projections()
    #     x = self.forward_features(x)
    #     x = self.head(x)
    #     return x


    def forward(self, x):

        if self.attn_type == "performer" and self.auto_check_redraw:
            self.check_redraw_projections()

        # convert to list
        if not isinstance(x, list):
            x = [x]
        # Perform forward pass separately on each resolution input.
        # The inputs corresponding to a single resolution are clubbed and single
        # forward is run on the same resolution inputs. Hence we do several
        # forward passes = number of different resolutions used. We then
        # concatenate all the output features.
        
        
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0)

        if self.use_dense_prediction:
            start_idx = 0
            
            for end_idx in idx_crops:
                _out_cls, _out_fea  = self.forward_features(torch.cat(x[start_idx: end_idx]))
                B, N, C = _out_fea.shape

                if start_idx == 0:
                    output_cls = _out_cls
                    output_fea = _out_fea.reshape(B * N, C)
                    npatch = [N]
                else:
                    output_cls = torch.cat((output_cls, _out_cls))
                    output_fea = torch.cat((output_fea, _out_fea.reshape(B * N, C) ))
                    npatch.append(N)
                start_idx = end_idx

            return self.head(output_cls), self.head_dense(output_fea), output_fea, npatch 

        else:

            start_idx = 0
            for end_idx in idx_crops:
                _out = self.forward_features(torch.cat(x[start_idx: end_idx]))
                if start_idx == 0:
                    output = _out
                else:
                    output = torch.cat((output, _out))
                start_idx = end_idx
            # Run the head forward on the concatenated features.
            return self.head(output)


@register_model
def get_cls_model(config, is_teacher=False, use_dense_prediction=False, **kwargs):
    msvit_spec = config.MODEL.SPEC
    args = dict(
        img_size=config.TRAIN.IMAGE_SIZE[0],
        drop_rate=msvit_spec.DROP,
        drop_path_rate= 0.0 if is_teacher else msvit_spec.DROP_PATH,
        norm_embed=msvit_spec.NORM_EMBED,
        avg_pool=msvit_spec.AVG_POOL,
    )
    args['arch'] = msvit_spec.MSVIT.ARCH
    args['sharew'] = msvit_spec.MSVIT.SHARE_W
    args['attn_type'] = msvit_spec.MSVIT.ATTN_TYPE
    args['share_kv'] = msvit_spec.MSVIT.SHARE_KV
    args['only_glo'] = msvit_spec.MSVIT.ONLY_GLOBAL
    args['sw_exact'] = msvit_spec.MSVIT.SW_EXACT
    args['ln_eps'] = msvit_spec.MSVIT.LN_EPS
    args['mode'] = msvit_spec.MSVIT.MODE
    args['pool_method'] = msvit_spec.MSVIT.POOL_METHOD

    args['use_dense_prediction'] = use_dense_prediction

    args['with_se'] = msvit_spec.MSVIT.WITH_SE
    args['se_mlp_ratio'] = msvit_spec.MSVIT.SE_MLP_RATIO
    args['se_mlp_balance'] = msvit_spec.MSVIT.SE_MLP_BALANCE

    msvit = MsViT(num_classes=config.MODEL.NUM_CLASSES, **args)

    model_mode = 'student' if is_teacher==False else 'teacher'
    rate = args['drop_path_rate']
    print(f'build vil {model_mode} with drop_path_rate {rate}')

    return msvit
