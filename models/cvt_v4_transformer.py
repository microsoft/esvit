import logging
import os
import torch
import torch.nn.functional as F
from functools import partial
from torch import nn, einsum
from torch._six import container_abcs

import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from timm.models.layers import DropPath, trunc_normal_

# helper methods
from .registry import register_model


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class PreNorm(nn.Module):
    def __init__(self, norm, dim, fn):
        super().__init__()
        self.norm = norm(dim)
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        x = rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        x = rearrange(x, 'b h w c -> b c h w')
        return self.fn(x, *args, **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, act_layer, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, int(dim * mult), 1),
            act_layer(),
            nn.Conv2d(int(dim * mult), dim, 1),
        )

    def forward(self, x):
        return self.net(x)


class DepthWiseConv2d(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        kernel_size,
        padding,
        stride,
        bias=True
    ):
        super().__init__()
        self.dw = nn.Conv2d(
            dim_in, dim_in,
            kernel_size=kernel_size,
            padding=padding,
            groups=dim_in,
            stride=stride,
            bias=False
        )
        self.bn = nn.BatchNorm2d(dim_in)
        self.pw = nn.Conv2d(
            dim_in, dim_out,
            kernel_size=1,
            bias=bias
        )

    def forward(self, x):
        x = self.dw(x)
        x = self.bn(x)
        x = self.pw(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        num_heads,
        qkv_bias,
        kernel_size,
        padding,
        window_size,
        shift_size,
        rel_pos_embed,
        **kwargs
    ):
        super().__init__()
        self.heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.scale = dim_out ** -0.5

        self.attend = nn.Softmax(dim=-1)

        self.qkv = DepthWiseConv2d(
            dim_in, dim_out*3, kernel_size,
            padding=padding, stride=1, bias=qkv_bias
        )

        self.proj_out = nn.Conv2d(dim_out, dim_in, 1)

        self.rel_pos_embed = rel_pos_embed
        if rel_pos_embed:
            self.init_rel_pos_embed(window_size, num_heads)

    def init_rel_pos_embed(self, window_size, num_heads):
        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        rel_pos_idx = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("rel_pos_idx", rel_pos_idx)

        # define a parameter table of relative position bias
        self.rel_pos_bias_table = nn.Parameter(
            torch.zeros(
                (2 * window_size - 1) * (2 * window_size - 1),
                num_heads
            )
        )  # 2*Wh-1 * 2*Ww-1, nH

        trunc_normal_(self.rel_pos_bias_table, std=.02)

    def forward(self, x, mask):
        shape = x.shape
        _, _, H, W, h = *shape, self.heads
        w = min(self.window_size, min(H, W))

        pad_l = pad_t = 0
        pad_r = (w - W % w) % w
        pad_b = (w - H % w) % w
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (pad_l, pad_r, pad_t, pad_b))
            _, _, Hp, Wp = x.shape
            s_x, s_y = Hp // w, Wp // w
        else:
            s_x, s_y = H // w, W // w

        q, k, v = self.qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(
                t, 'b (h d) (s_x w_x) (s_y w_y) -> (b s_x s_y) h (w_x w_y) d',
                h=h, s_x=s_x, s_y=s_y, w_x=w, w_y=w
            ),
            (q, k, v)
        )

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        if self.rel_pos_embed:
            rel_pos_bias = self.rel_pos_bias_table[self.rel_pos_idx.view(-1)]\
                               .view(
                                   self.window_size * self.window_size,
                                   self.window_size * self.window_size,
                                   -1
                               )  # Wh*Ww,Wh*Ww,nH
            rel_pos_bias = rel_pos_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            dots = dots + rel_pos_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            B_, H, N, M = dots.shape
            dots = dots.view(
                B_ // nW, nW, self.heads, N, M
            ) + mask.unsqueeze(1).unsqueeze(0)
            dots = dots.view(-1, self.heads, N, M)

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(
            out, '(b s_x s_y) h (w_x w_y) d -> b (h d) (s_x w_x) (s_y w_y)',
            h=h, s_x=s_x, s_y=s_y, w_x=w, w_y=w
        ).contiguous()

        if pad_r > 0 or pad_b > 0:
            out = out[:, :, :H, :W].contiguous()

        return self.proj_out(out)

    @staticmethod
    def compute_macs(module, input, output):
        # T: num_token
        # S: num_token
        input = input[0]
        B, C, H, W = input.shape

        flops = 0
        params = sum([p.numel() for p in module.qkv.dw.parameters()])
        flops += params * H * W
        params = sum([p.numel() for p in module.qkv.pw.parameters()])
        flops += params * H * W
        params = sum([p.numel() for p in module.proj_out.parameters()])
        flops += params * H * W

        flops += 2 * C * H * W * module.window_size ** 2

        module.__flops__ += flops


class Transformer(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.,
        qkv_bias=False,
        drop_path_rate=None,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        kernel_qkv=3,
        padding_qkv=1,
        window_size=-1,
        shift=False,
        rel_pos_embed=False,
        **kwargs
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for i in range(depth):
            shift_size = window_size//2 if shift and i % 2 == 1 else 0,
            self.layers.append(nn.ModuleList([
                PreNorm(
                    norm_layer, embed_dim,
                    Attention(
                        dim_in=embed_dim,
                        dim_out=embed_dim,
                        num_heads=num_heads,
                        qkv_bias=qkv_bias,
                        kernel_size=kernel_qkv,
                        padding=padding_qkv,
                        window_size=window_size,
                        shift_size=shift_size,
                        rel_pos_embed=rel_pos_embed,
                        *kwargs
                    )
                ),
                PreNorm(
                    norm_layer, embed_dim,
                    FeedForward(embed_dim, act_layer, mlp_ratio)
                ),
                DropPath(drop_path_rate[i])
                if isinstance(drop_path_rate, list) else nn.Identity()
            ]))

        self.window_size = window_size
        self.shift = shift

    def build_attn_mask(self, x):
        _, _, H, W = x.shape
        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
        shift_size = self.window_size//2
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -shift_size),
            slice(-shift_size, None)
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -shift_size),
            slice(-shift_size, None)
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        s_x = Hp // self.window_size
        s_y = Wp // self.window_size
        mask_windows = rearrange(
            img_mask, 'i (s_x w_x) (s_y w_y) j -> (i s_x s_y) w_x w_y j',
            s_x=s_x, s_y=s_y, w_y=self.window_size, w_x=self.window_size
        )
        # mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(
            -1, self.window_size * self.window_size
        )
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(
            attn_mask != 0, float(-100.0)
        ).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x):
        attn_mask = self.build_attn_mask(x) if self.shift else None
        for attn, ff, drop_path in self.layers:
            x = drop_path(attn(x, attn_mask)) + x
            x = drop_path(ff(x)) + x
        return x

    def forward_with_features(self, x):
        attn_mask = self.build_attn_mask(x) if self.shift else None
        
        feats = []
        for attn, ff, drop_path in self.layers:
            x = drop_path(attn(x, attn_mask)) + x
            x = drop_path(ff(x)) + x
            feats.append(x)
        return x, feats


class ConvEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(
        self,
        patch_size=7,
        in_chans=3,
        embed_dim=64,
        stride=4,
        padding=2,
        norm_layer=None
    ):
        super().__init__()
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=padding
        )
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        x = self.proj(x)

        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        if self.norm:
            x = self.norm(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W).contiguous()

        return x


class ResStem(nn.Module):
    def __init__(self, channels_stem, deep=False):
        super().__init__()
        if deep:
            self.stem = nn.Sequential(
                nn.Conv2d(
                    3, channels_stem, kernel_size=3, stride=2, padding=1,
                    bias=False
                ),
                nn.BatchNorm2d(channels_stem),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    channels_stem, channels_stem,
                    kernel_size=3, stride=1,
                    padding=1, bias=False
                ),
                nn.BatchNorm2d(channels_stem),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    channels_stem, channels_stem,
                    kernel_size=3, stride=2,
                    padding=1, bias=False
                ),
                nn.BatchNorm2d(channels_stem),
                nn.ReLU(inplace=True)
            )
        else:
            self.stem = nn.Sequential(
                nn.Conv2d(
                    3, channels_stem, kernel_size=3, stride=2,
                    padding=1, bias=False
                ),
                nn.BatchNorm2d(channels_stem),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    channels_stem, channels_stem,
                    kernel_size=3, stride=2,
                    padding=1, bias=False
                ),
                nn.BatchNorm2d(channels_stem),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        x = self.stem(x)

        return x


class CvT(nn.Module):
    def __init__(
        self,
        *,
        num_classes,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        init='trunc_norm',
        use_dense_prediction=False,
        spec=None
    ):
        super().__init__()

        self.num_stages = spec['NUM_STAGES']

        total_depth = sum(spec['DEPTH'])
        logging.info(f'=> total path: {total_depth}')
        drop_path_rate = spec['DROP_PATH_RATE']
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ]  # stochastic depth decay rule
        in_chans = 3
        depth_accum=0
        for i in range(self.num_stages):
            kwargs = {
                'patch_size': spec['PATCH_SIZE'][i],
                'patch_stride': spec['PATCH_STRIDE'][i],
                'patch_padding': spec['PATCH_PADDING'][i],
                'embed_dim': spec['DIM_EMBED'][i],
                'depth': spec['DEPTH'][i],
                'num_heads': spec['NUM_HEADS'][i],
                'mlp_ratio': spec['MLP_RATIO'][i],
                'qkv_bias': spec['QKV_BIAS'][i],
                'kernel_qkv': spec['KERNEL_QKV'][i],
                'padding_qkv': spec['PADDING_QKV'][i],
                'window_size': spec['WINDOW_SIZE'][i],
                'shift': spec['SHIFT'][i],
            }

            if i == 0 and getattr(spec, 'RES_STEM', False):
                conv = ResStem(kwargs['embed_dim'], True)
            else:
                conv = ConvEmbed(
                    patch_size=kwargs['patch_size'],
                    in_chans=in_chans,
                    embed_dim=kwargs['embed_dim'],
                    stride=kwargs['patch_stride'],
                    padding=kwargs['patch_padding'],
                    norm_layer=norm_layer
                )

            stage = nn.Sequential(
                conv,
                Transformer(
                    embed_dim=kwargs['embed_dim'],
                    depth=kwargs['depth'],
                    num_heads=kwargs['num_heads'],
                    mlp_ratio=kwargs['mlp_ratio'],
                    qkv_bias=kwargs['qkv_bias'],
                    drop_path_rate=dpr[
                        depth_accum: depth_accum+kwargs['depth']
                    ],
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    kernel_qkv=kwargs['kernel_qkv'],
                    padding_qkv=kwargs['padding_qkv'],
                    window_size=kwargs['window_size'],
                    shift=kwargs['shift'],
                    rel_pos_embed=spec['REL_POS_EMBED']
                )
            )
            setattr(self, f'stage{i}', stage)
            in_chans = spec['DIM_EMBED'][i]
            depth_accum += kwargs['depth']

        self.norm = norm_layer(in_chans)
        self.avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Rearrange('... () () -> ...')
        )
        self.head = nn.Linear(in_chans, num_classes) if num_classes > 0 else nn.Identity()

        # Region prediction head
        self.use_dense_prediction = use_dense_prediction
        if self.use_dense_prediction: self.head_dense = None


        if init == 'xavier':
            self.apply(self._init_weights_xavier)
        else:
            self.apply(self._init_weights_trunc_normal)

    def _init_weights_trunc_normal(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            logging.info('=> init weight from trunc norm')
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                logging.info('=> init bias to zeros')
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _init_weights_xavier(self, m):
        if isinstance(m, nn.Linear):
            logging.info('=> init weight of Linear from xavier uniform')
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                logging.info('=> init bias of Linear to zeros')
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward_features(self, x):
        for i in range(self.num_stages):
            x = getattr(self, f'stage{i}')(x)

        H, W = x.shape[-2], x.shape[-1]
        x = rearrange(x, 'b c h w -> b (h w) c')
        x_region = self.norm(x)
        x = rearrange(x_region, 'b (h w) c -> b c h w', h=H, w=W)

        x = self.avg_pool(x)  # B C 1

        if self.use_dense_prediction:
            return x, x_region
        else:
            return x

        

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

        for i in range(self.num_stages):
            stage = getattr(self, f'stage{i}')
            
            x = stage[0](x)
            x, fea = stage[1].forward_with_features(x)
            # x = getattr(self, f'stage{i}')(x)

            # print(f'fea list length {len(fea)}')

        # for i, layer in enumerate(self.layers):
        #     x, fea = layer.forward_with_features(x)

            if i >= start_stage:
                for x_ in fea[start_blk:]:

                    # print(f'x_ shape {x_.shape}')
                    if i == self.num_stages-1: # use the norm in the last stage
                        x_ = rearrange(x_, 'b c h w -> b h w c').contiguous()
                        x_ = self.norm(x_)
                        x_ = rearrange(x_, 'b h w c -> b c h w').contiguous()


                    x_avg = torch.flatten(self.avg_pool(x_), 1)  # B C     
                    # print(f'Stage {i},  x_avg {x_avg.shape},  x_ {x_.shape}')  
 
                    output.append(x_avg)

                start_blk = 0

        return torch.cat(output, dim=-1)



    def forward(self, x):

        # convert to list
        if not isinstance(x, list):
            x = [x]

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

    def init_weights(self, pretrained='', pretrained_layers=[], verbose=True):
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained, map_location='cpu')
            logging.info(f'=> loading pretrained model {pretrained}')
            model_dict = self.state_dict()
            pretrained_dict = {
                k: v for k, v in pretrained_dict.items()
                if k in model_dict.keys()
            }
            need_init_state_dict = {}
            for k, v in pretrained_dict.items():
                need_init = (
                        k.split('.')[0] in pretrained_layers
                        or pretrained_layers[0] is '*'
                )
                if need_init:
                    if verbose:
                        logging.info(f'=> init {k} from {pretrained}')
                    need_init_state_dict[k] = v
            self.load_state_dict(need_init_state_dict, strict=False)


@register_model
def get_cls_model(config, is_teacher=False, use_dense_prediction=False, **kwargs):
    cvt_spec = config.MODEL.SPEC

    if is_teacher: cvt_spec['DROP_PATH_RATE']=0.0

    cvt = CvT(
        num_classes=config.MODEL.NUM_CLASSES,
        act_layer=QuickGELU,
        norm_layer=partial(LayerNorm, eps=1e-5),
        init='trunc_norm',
        use_dense_prediction=use_dense_prediction,
        spec=cvt_spec
    )

    if config.MODEL.INIT_WEIGHTS:
        cvt.init_weights(
            config.MODEL.PRETRAINED,
            config.MODEL.PRETRAINED_LAYERS,
            config.VERBOSE
        )

    return cvt
