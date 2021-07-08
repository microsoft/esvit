
from functools import lru_cache
import torch
from torch import einsum
from torch.cuda.amp import autocast


class SlidingChunk2D(torch.autograd.Function):
    """
    Class to encapsulate for sliding chunk implementation of vision longformer
    """
    mode_dict = {
        1: (1, 1),  # -1, -1
        2: (1, 0),  # -1, 0
        3: (1, -1),  # -1, 1
        4: (0, 1),  # 0, -1
        5: (0, -1),  # 0, 1
        6: (-1, 1),  # 1, -1
        7: (-1, 0),  # 1, 0
        8: (-1, -1),  # 1, 1
    }

    @staticmethod
    def slidingchunk_qk(q_img: torch.Tensor, k_img: torch.Tensor, mode: int):
        '''
        q_img x k_img = attn11 ==> Useful for query x key = attention_scores
        The cyclic padding strategy
        q_img, k_img: (B * H, M, mx, my, W**2)
        attn11： (B*H, mx, my, W**2, 9*W**2), mode=0
                (B*H, mx, my, W**2, W**2), mode=-1
                (B*H, mx, my, W**2, 2*W**2), mode=i>0
        mode: 0 -> full, -1 -> only self, i (>0) -> self+block_i
        '''
        if mode == 0:
            return torch.cat([
                # -1, -1
                einsum('b c m n l, b c m n t -> b m n l t', q_img,
                       torch.roll(k_img, shifts=(1, 1), dims=(2, 3))),
                # -1, 0
                einsum('b c m n l, b c m n t -> b m n l t', q_img,
                       torch.roll(k_img, shifts=1, dims=2)),
                # -1, 1
                einsum('b c m n l, b c m n t -> b m n l t', q_img,
                       torch.roll(k_img, shifts=(1, -1), dims=(2, 3))),
                # 0, -1
                einsum('b c m n l, b c m n t -> b m n l t', q_img,
                       torch.roll(k_img, shifts=1, dims=3)),
                # 0, 0
                einsum('b c m n l, b c m n t -> b m n l t', q_img,
                       k_img),
                # 0, 1
                einsum('b c m n l, b c m n t -> b m n l t', q_img,
                       torch.roll(k_img, shifts=-1, dims=3)),
                # 1, -1
                einsum('b c m n l, b c m n t -> b m n l t', q_img,
                       torch.roll(k_img, shifts=(-1, 1), dims=(2, 3))),
                # 1, 0
                einsum('b c m n l, b c m n t -> b m n l t', q_img,
                       torch.roll(k_img, shifts=-1, dims=2)),
                # 1, 1
                einsum('b c m n l, b c m n t -> b m n l t', q_img,
                       torch.roll(k_img, shifts=(-1, -1), dims=(2, 3))),
            ], dim=-1)
        elif mode == -1:
            return einsum(
                'b c m n l, b c m n t -> b m n l t', q_img, k_img
            ) * 1.0
        else:
            shift = SlidingChunk2D.mode_dict[mode]
            return torch.cat([
                # 0, 0
                einsum('b c m n l, b c m n t -> b m n l t', q_img, k_img),
                # x, x
                einsum('b c m n l, b c m n t -> b m n l t', q_img,
                       torch.roll(k_img, shifts=shift, dims=(2, 3))),
            ], dim=-1)


    @staticmethod
    def slidingchunk_av(attn: torch.Tensor, v_img: torch.Tensor, mode: int):
        '''
        attn x v_img = x ==> Useful for attn x value = context
        The cyclic padding strategy
        v_img, context: (B * H, M, mx, my, W**2)
        attn： (B*H, mx, my, W**2, 9*W**2), mode=0
                (B*H, mx, my, W**2, W**2), mode=-1
                (B*H, mx, my, W**2, 2*W**2), mode=i>0
        mode: 0 -> full, -1 -> only self, i (>0) -> self+block_i
        '''
        w2 = v_img.shape[-1]
        if mode == 0:
            attnn1n1, attnn10, attnn11, attn0n1, attn00, attn01, attn1n1, attn10, attn11 = torch.split(
                attn, w2, dim=-1
            )
        elif mode == -1:
            attn00 = attn
        else:
            attn00, attnxx = torch.split(
                attn, w2, dim=-1
            )
        output = einsum('b m n l t, b c m n t -> b c m n l', attn00, v_img)  # 0,0

        if mode == 0:
            output = output + einsum('b m n l t, b c m n t -> b c m n l', attnn1n1,
                                     torch.roll(v_img, shifts=(1, 1), dims=(2, 3)))  # -1,-1
            output = output + einsum('b m n l t, b c m n t -> b c m n l', attnn10,
                                     torch.roll(v_img, shifts=1, dims=2))  # -1,0
            output = output + einsum('b m n l t, b c m n t -> b c m n l', attnn11,
                                     torch.roll(v_img, shifts=(1, -1), dims=(2, 3)))  # -1,1
            output = output + einsum('b m n l t, b c m n t -> b c m n l', attn0n1,
                                     torch.roll(v_img, shifts=1, dims=3))  # 0,-1
            output = output + einsum('b m n l t, b c m n t -> b c m n l', attn01,
                                     torch.roll(v_img, shifts=-1, dims=3))  # 0,1
            output = output + einsum('b m n l t, b c m n t -> b c m n l', attn1n1,
                                     torch.roll(v_img, shifts=(-1, 1), dims=(2, 3)))  # 1,-1
            output = output + einsum('b m n l t, b c m n t -> b c m n l', attn10,
                                     torch.roll(v_img, shifts=-1, dims=2))  # 1,0
            output = output + einsum('b m n l t, b c m n t -> b c m n l', attn11,
                                     torch.roll(v_img, shifts=(-1, -1), dims=(2, 3)))  # 1,1
        elif mode > 0:
            shift = SlidingChunk2D.mode_dict[mode]
            output = output + einsum('b m n l t, b c m n t -> b c m n l', attnxx,
                                     torch.roll(v_img, shifts=shift, dims=(2, 3)))  # 1,1
        else:
            output = output * 1.0

        return output

    @staticmethod
    def slidingchunk_agrad(attn: torch.Tensor, grad_x: torch.Tensor, mode: int):
        '''
        attn.t() x grad_x = grad_v ==> Useful for attn.t() x grad_x = grad_v
        The cyclic padding strategy
        grad_x, grad_v: (B * H, M, mx, my, W**2)
        attn： (B*H, mx, my, W**2, 9*W**2), mode=0
                (B*H, mx, my, W**2, W**2), mode=-1
                (B*H, mx, my, W**2, 2*W**2), mode=i>0
        mode: 0 -> full, -1 -> only self, i (>0) -> self+block_i
        '''
        w2 = grad_x.shape[-1]
        if mode == 0:
            attnn1n1, attnn10, attnn11, attn0n1, attn00, attn01, attn1n1, attn10, attn11 = torch.split(
                attn, w2, dim=-1
            )
        elif mode == -1:
            attn00 = attn
        else:
            attn00, attnxx = torch.split(
                attn, w2, dim=-1
            )

        # 0,0
        output = einsum('b m n l t, b c m n l -> b c m n t', attn00, grad_x)

        if mode == 0:
            # -1,-1
            output = output + torch.roll(
                einsum('b m n l t, b c m n l -> b c m n t', attnn1n1, grad_x),
                shifts=(-1, -1), dims=(2, 3))
            # -1,0
            output = output + torch.roll(
                einsum('b m n l t, b c m n l -> b c m n t', attnn10, grad_x),
                shifts=-1, dims=2)
            # -1,1
            output = output + torch.roll(
                einsum('b m n l t, b c m n l -> b c m n t', attnn11, grad_x),
                shifts=(-1, 1), dims=(2, 3))
            # 0,-1
            output = output + torch.roll(
                einsum('b m n l t, b c m n l -> b c m n t', attn0n1, grad_x),
                shifts=-1, dims=3)
            # 0,1
            output = output + torch.roll(
                einsum('b m n l t, b c m n l -> b c m n t', attn01, grad_x),
                shifts=1, dims=3)
            # 1,-1
            output = output + torch.roll(
                einsum('b m n l t, b c m n l -> b c m n t', attn1n1, grad_x),
                shifts=(1, -1), dims=(2, 3))
            # 1,0
            output = output + torch.roll(
                einsum('b m n l t, b c m n l -> b c m n t', attn10, grad_x),
                shifts=1, dims=2)
            # 1,1
            output = output + torch.roll(
                einsum('b m n l t, b c m n l -> b c m n t', attn11, grad_x),
                shifts=(1, 1), dims=(2, 3))
        elif mode > 0:
            shift = SlidingChunk2D.mode_dict[mode]
            shift = (-shift[0], -shift[1])
            output = output + torch.roll(
                einsum('b m n l t, b c m n l -> b c m n t', attnxx, grad_x),
                shifts=shift, dims=(2, 3))
        else:
            output = output * 1.0

        return output

    @staticmethod
    @autocast()  # comment this out if AMP is not used
    def forward(ctx, t1: torch.Tensor, t2: torch.Tensor,
                is_t1_diagonaled: bool = False, mode: int = 0) -> torch.Tensor:
        """Compuates sliding chunk mm of t1 and t2.
        args:
        t1: torch.Tensor = (B * H, M, mx, my, W**2) if is_t1_diagonaled = false,
                         = (B*H, mx, my, W**2, 9*W**2) if is_t1_diagonaled = true, mode=0.
                         = (B*H, mx, my, W**2, W**2) if is_t1_diagonaled = true, mode=-1.
                         = (B*H, mx, my, W**2, 2*W**2) if is_t1_diagonaled = true, mode=i>0.
        t2: torch.Tensor = (B * H, M, mx, my, W**2). This is always a
            non-diagonaled tensor, e.g. `key_layer` or `value_layer`
        is_t1_diagonaled: is t1 a diagonaled or a regular tensor
        mode: 0 -> full, -1 -> only self, i (>0) -> self+block_i
        returns:
        is_t1_diagonaled = true:
        torch.Tensor = (B * H, M, mx, my, W**2)
        mode=0, is_t1_diagonaled = false:
        torch.Tensor = (B*H, mx, my, W**2, 9*W**2)
        mode=-1, is_t1_diagonaled = false:
        torch.Tensor = (B*H, mx, my, W**2, W**2)
        mode=i>0, is_t1_diagonaled = false:
        torch.Tensor = (B*H, mx, my, W**2, W**2)
        """
        ctx.save_for_backward(t1, t2)
        ctx.is_t1_diagonaled = is_t1_diagonaled
        ctx.mode = mode
        if is_t1_diagonaled:
            return SlidingChunk2D.slidingchunk_av(t1, t2, mode)
        else:
            return SlidingChunk2D.slidingchunk_qk(t1, t2, mode)

    @staticmethod
    @autocast()  # comment this out if AMP is not used
    def backward(ctx, grad_output):
        t1, t2 = ctx.saved_tensors
        is_t1_diagonaled = ctx.is_t1_diagonaled
        mode = ctx.mode
        if is_t1_diagonaled:
            grad_t1 = SlidingChunk2D.slidingchunk_qk(grad_output, t2, mode)
            grad_t2 = SlidingChunk2D.slidingchunk_agrad(t1, grad_output, mode)
        else:
            grad_t1 = SlidingChunk2D.slidingchunk_av(grad_output, t2, mode)
            grad_t2 = SlidingChunk2D.slidingchunk_agrad(grad_output, t1, mode)
        return grad_t1, grad_t2, None, None


@lru_cache()
def _get_invalid_locations_mask_cyclic(nx: int, ny: int, padx: int, pady: int,
                                w: int, device: str):
    w2 = w ** 2
    mask = torch.BoolTensor([
        [
            (i // ny + (j // w2) // 3 == nx and
             (nx - 1) * w + (j % w2) // w >= nx * w - padx) or
            (i % ny + (j // w2) % 3 == ny and
             (ny - 1) * w + (j % w2) % w >= ny * w - pady)
            for j in range(9 * w2)
        ]
        for i in range(nx * ny)
    ], device='cpu')

    # We should count the w2 in the query here
    num_invalid = w2 * mask.sum()

    return mask.to(device), num_invalid.to(device)


@lru_cache()
def _get_invalid_locations_mask_zero(nx: int, ny: int, padx: int, pady: int,
                                w: int, device: str):
    w2 = w ** 2
    mask = torch.BoolTensor([
        [
            i // ny + (j // w2) // 3 - 1 < 0 or
            i // ny + (j // w2) // 3 - 1 >= nx or
            (i // ny + (j // w2) // 3 - 1) * w + (j % w2) // w >= nx * w - padx or
            i % ny + (j // w2) % 3 - 1 < 0 or
            i % ny + (j // w2) % 3 - 1 >= ny or
            (i % ny + (j // w2) % 3 - 1) * w + (j % w2) % w >= ny * w - pady
            for j in range(9 * w2)
        ]
        for i in range(nx * ny)
    ], device='cpu')

    # We should count the w2 in the query here
    num_invalid = w2 * mask.sum()

    return mask.to(device), num_invalid.to(device)


@lru_cache()
def _get_invalid_locations_mask_exact(nx: int, ny: int, padx: int, pady: int,
                                      w: int, device: str):
    w2 = w ** 2
    nx_max = nx * w - 1 - padx
    ny_max = ny * w - 1 - pady
    mask = torch.BoolTensor([
        [
            [
                (i // ny + (j // w2) // 3 - 1) * w + (j % w2) // w < max(0, (
                        i // ny - 1) * w + l // w) or
                (i // ny + (j // w2) // 3 - 1) * w + (j % w2) // w > min(
                    nx_max, (i // ny + 1) * w + l // w) or
                (i % ny + (j // w2) % 3 - 1) * w + (j % w2) % w < max(0, (
                        i % ny - 1) * w + l % w) or
                (i % ny + (j // w2) % 3 - 1) * w + (j % w2) % w > min(
                    ny_max, (i % ny + 1) * w + l % w)
                for j in range(9 * w2)
            ]
            for l in range(w2)
        ]
        for i in range(nx * ny)
    ], device='cpu')
    num_invalid = mask.sum()

    return mask.to(device), num_invalid.to(device)


def mask_invalid_locations(input_tensor: torch.Tensor, nx: int, ny: int,
                           padx: int, pady: int, w: int,
                           exact: int, mode: int = 0) -> torch.Tensor:
    """exact
    1: exact sliding window
    0: blockwise sliding chunk with zero padding
    -1: blockwise sliding chunk with cyclic padding
    mode: 0 -> full, -1 -> only self, i (>0) -> self+block_i
    """
    w2 = w ** 2
    if exact == 1 and mode == 0:
        mask, num_invalid = _get_invalid_locations_mask_exact(
            nx, ny, padx, pady, w, input_tensor.device)
        mask = mask.view(1, nx, ny, w2, -1).expand(input_tensor.size())
    else:
        if exact == 0:
            mask, num_invalid = _get_invalid_locations_mask_zero(
                nx, ny, padx, pady, w, input_tensor.device)
        elif exact == -1:
            mask, num_invalid = _get_invalid_locations_mask_cyclic(
                nx, ny, padx, pady, w, input_tensor.device)
        else:
            raise ValueError("longsc exact should be in [0,1,-1]!")
        if mode == -1:
            mask = mask[:, 4 * w2:5 * w2]
            num_invalid = w2 * mask.sum()
        elif mode > 0:
            chunk_id = mode if mode > 4 else mode - 1
            mask = torch.cat([
                mask[:, 4 * w2:5 * w2],
                mask[:, chunk_id * w2:(chunk_id+1) * w2],
            ], dim=-1)
            num_invalid = w2 * mask.sum()
        mask = mask.view(1, nx, ny, 1, -1).expand(input_tensor.size())
    input_tensor.masked_fill_(mask, -float('inf'))

    return num_invalid


def slidingchunk_2dautograd(t1: torch.Tensor, t2: torch.Tensor,
                is_t1_diagonaled: bool = False, mode: int = 0) -> torch.Tensor:
    if is_t1_diagonaled:
        return SlidingChunk2D.slidingchunk_av(t1, t2, mode)
    else:
        return SlidingChunk2D.slidingchunk_qk(t1, t2, mode)


slidingchunk_2d = SlidingChunk2D.apply
