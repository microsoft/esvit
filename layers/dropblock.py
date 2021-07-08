import logging

import torch


class DropBlock(torch.nn.Module):
    def __init__(self,
                 block_size=7,
                 keep_prob=0.9,
                 current_step=0.,
                 train_steps=1.):
        super(DropBlock, self).__init__()
        self.block_size = block_size
        self.keep_prob = keep_prob
        self.current_step = current_step
        self.train_steps = train_steps

        logging.info('build dropblock: block size({}), keep prob({})'.format(
            block_size, keep_prob))

    def init(self, current_step, train_steps):
        self.current_step = current_step
        self.train_steps = train_steps

    def step(self):
        self.current_step += 1

    def forward(self, x):
        current_ratio = self.current_step / self.train_steps
        _keep_prob = (1 - current_ratio * (1 - self.keep_prob))
        if not self.training or _keep_prob == 1.0:
            return x

        nb, nc, height, width = x.size()
        if height != width:
            raise ValueError('Input tensor with width != height is not supported.')

        block_size = min(self.block_size, width)
        gamma = (1. - _keep_prob) * width**2 / block_size**2 / (
            width - block_size + 1)**2

        w_i, h_i = torch.meshgrid(
            torch.arange(width, device=x.device),
            torch.arange(height, device=x.device)
        )
        valid_block_center = torch.logical_and(
            torch.logical_and(w_i >= int(block_size // 2),
                              w_i < width - (block_size - 1) // 2),
            torch.logical_and(h_i >= int(block_size // 2),
                              h_i < width - (block_size - 1) // 2)
        )
        valid_block_center.unsqueeze_(0)
        valid_block_center.unsqueeze_(0)

        randnoise = torch.rand(x.size(), device=x.device)

        block_pattern = (
            1 - valid_block_center.to(torch.float32) + (1 - gamma) + randnoise
        ) >= 1
        block_pattern = block_pattern.to(torch.float32).cuda()

        if block_size == width:
            block_pattern, _ = block_pattern.reshape((nb, nc, -1)).min(
                dim=-1, keepdim=True
            )
            block_pattern.unsqueeze_(-1)
        else:
            ksize = (block_size, block_size)
            padding = (block_size // 2, block_size // 2)
            stride = (1, 1)
            block_pattern = -torch.nn.functional.max_pool2d(
                -block_pattern, ksize, stride, padding
            )

        return block_pattern * x * (block_pattern.numel()/block_pattern.sum())
