import logging

import torch
import torch.functional as F
import torch.nn as nn


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True, h_max=1):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
        self.h_max = h_max

    def forward(self, x):
        return self.relu(x + 3) * self.h_max / 6


class DYReLU2(nn.Module):
    def __init__(self, inp, oup, reduction=4, lambda_a=1.0, K2=True,
                 use_bias=True, init_a=[1.0, 0.0], init_b=[0.0, 0.0]):
        super(DYReLU2, self).__init__()
        self.oup = oup
        self.lambda_a = lambda_a * 2
        self.K2 = K2
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.use_bias = use_bias
        if K2:
            self.exp = 4 if use_bias else 2
        else:
            self.exp = 2 if use_bias else 1
        self.init_a = init_a
        self.init_b = init_b

        # determine squeeze
        if reduction == 4:
            squeeze = inp // reduction
        else:
            squeeze = _make_divisible(inp // reduction, 4)
        logging.info('=> reduction: {}, squeeze: {}/{}'.format(reduction, inp, squeeze))
        logging.info('=> init_a: {}, init_b: {}'.format(self.init_a, self.init_b))
 
        self.fc = nn.Sequential(
                nn.Linear(inp, squeeze),
                nn.ReLU(inplace=True),
                nn.Linear(squeeze, oup*self.exp),
                h_sigmoid()
        )

    def forward(self, x):
        if isinstance(x, list):
            x_in = x[0]
            x_out = x[1]
        else:
            x_in = x
            x_out = x
        b, c, _, _ = x_in.size()
        y = self.avg_pool(x_in).view(b, c)
        y = self.fc(y).view(b, self.oup*self.exp, 1, 1)

        if self.exp == 4:
            a1, b1, a2, b2 = torch.split(y, self.oup, dim=1)
            a1 = (a1 - 0.5) * self.lambda_a + self.init_a[0]#1.0
            a2 = (a2 - 0.5) * self.lambda_a + self.init_a[1]
     
            b1 = b1 - 0.5 + self.init_b[0]
            b2 = b2 - 0.5 + self.init_b[1]
            out = torch.max(x_out * a1 + b1, x_out * a2 + b2)
        elif self.exp == 2:
            if self.use_bias: # bias but not PL
                a1, b1 = torch.split(y, self.oup, dim=1)
                a1 = (a1 - 0.5) * self.lambda_a + self.init_a[0]# 1.0
                b1 = b1 - 0.5 + self.init_b[0]
                out = x_out * a1 + b1

            else:
                a1, a2 = torch.split(y, self.oup, dim=1)
                a1 = (a1 - 0.5) * self.lambda_a + self.init_a[0] #1.0
                a2 = (a2 - 0.5) * self.lambda_a + self.init_a[1]
                out = torch.max(x_out * a1, x_out * a2)
               
        elif self.exp == 1:
                a1 = y
                a1 = (a1 - 0.5) * self.lambda_a + self.init_a[0] #1.0
                out = x_out * a1

        return out


class DYReLUSpaAtt2(nn.Module):
    def __init__(self, inp, oup, reduction=4, lambda_a=1.0, K2=True,
                 use_bias=True, spa_tau=10.0):
        super(DYReLUSpaAtt2, self).__init__()
        self.oup = oup
        self.lambda_a = lambda_a * 2
        self.K2 = K2
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.tau = spa_tau

        self.use_bias = use_bias
        if K2:
            self.exp = 4 if use_bias else 2
        else:
            self.exp = 2 if use_bias else 1
 
        self.fc = nn.Sequential(
                nn.Linear(inp, inp // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(inp // reduction, oup*self.exp),
                h_sigmoid()
        )

        self.spa = nn.Sequential(
            nn.Conv2d(inp, 1, kernel_size=1),
            nn.BatchNorm2d(1),
        )

    def forward(self, x):
        if isinstance(x, list):
            x_in = x[0]
            x_out = x[1]
        else:
            x_in = x
            x_out = x
        b, c, h, w = x_in.size()
        y = self.avg_pool(x_in).view(b, c)
        y = self.fc(y).view(b, self.oup*self.exp, 1, 1)

        if self.exp == 4:
            a1, b1, a2, b2 = torch.split(y, self.oup, dim=1)
    #        a1 = a1 * 4 - 1.0
    #        a2 = a2 * 4 - 2.0
            a1 = (a1 - 0.5) * self.lambda_a + 1.0
            a2 = (a2 - 0.5) * self.lambda_a
     
            b1 = b1 - 0.5
            b2 = b2 - 0.5
            out = torch.max(x_out * a1 + b1, x_out * a2 + b2)
        elif self.exp == 2:
            if self.use_bias: # bias but not PL
                a1, b1 = torch.split(y, self.oup, dim=1)
                a1 = (a1 - 0.5) * self.lambda_a + 1.0
                b1 = b1 - 0.5
                out = x_out * a1 + b1

            else:
                a1, a2 = torch.split(y, self.oup, dim=1)
                a1 = (a1 - 0.5) * self.lambda_a + 1.0
                a2 = (a2 - 0.5) * self.lambda_a
                out = torch.max(x_out * a1, x_out * a2)
               
        elif self.exp == 1:
                a1 = y
                a1 = (a1 - 0.5) * self.lambda_a + 1.0
                out = x_out * a1
        # add spational attention
        ys = self.spa(x_in).view(b, -1)
        ys = F.softmax(ys/self.tau, dim=1).view(b, 1, h, w) * h * w
        ys = F.hardtanh(ys, 0, 3, inplace=True)/3

        return out * ys
