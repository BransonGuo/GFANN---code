import torch
from torch import nn
import numpy as np
from collections import OrderedDict
from .model_lib_soft_PCA import AdditiveLayer
import math


class FactorAugmentedNN(nn.Module):
    def __init__(self, p, r_bar, depth, width, dp_matrix, fix_dp_mat=True, input_dropout=False, dropout_rate=0.0, check_depth=False, **kwargs):
        super(FactorAugmentedNN, self).__init__()
        self.use_input_dropout = input_dropout
        self.input_dropout = nn.Dropout(p=dropout_rate)
        self.diversified_projection = nn.Linear(p, r_bar, bias=False)
        self.batch_norm = nn.BatchNorm1d(p)
        dp_matrix_tensor = torch.tensor(
            np.transpose(dp_matrix), dtype=torch.float32)
        self.diversified_projection.weight = nn.Parameter(
            dp_matrix_tensor, requires_grad=not fix_dp_mat)

        relu_nn = [('linear1', nn.Linear(r_bar, width)),
                   ('relu1', nn.LeakyReLU(0.1))]
        # when depth is 3, there will be projection, linear1, + 2 layers, output: projection + 3 hidden
        for i in range(2, depth + 1):
            relu_nn.append(('linear{}'.format(i), nn.Linear(width, width)))
            relu_nn.append(('relu{}'.format(i), nn.LeakyReLU(0.1)))

        relu_nn.append(('linear{}'.format(depth+1), nn.Linear(width, 1)))
        self.relu_stack = nn.Sequential(
            OrderedDict(relu_nn)
        )

    def forward(self, x, is_training=False, **kwargs):
        if self.use_input_dropout and is_training:
            x = self.input_dropout(x)
        x = self.diversified_projection(x)
        pred = self.relu_stack(x)
        return pred


class RegressionNN_additive(nn.Module):
    def __init__(self, p, depth, width, r_bar, add_width, kernel_width=1, add_depth=2, tail_depth=2, input_dropout=False, dropout_rate=0.0, check_depth=False, **kwargs):
        super().__init__()
        self.use_input_dropout = input_dropout
        self.input_dropout = nn.Dropout(p=dropout_rate)
        self.batch_norm = nn.BatchNorm1d(p)
        if check_depth:
            assert depth >= 3
        add_nn = [('pre_add_linear1', nn.Linear(p, r_bar)),
                  ('pre_add_relu1', nn.LeakyReLU(0.1))]
        add_nn.extend([('add1', AdditiveLayer([kernel_width] * r_bar,
                      [add_width] * r_bar)), ('add_relu1', nn.LeakyReLU(0.1))])
        for i in range(2, add_depth + 1):
            add_nn.append(('add{}'.format(i), AdditiveLayer(
                [add_width] * r_bar, [add_width] * r_bar)))
            add_nn.append(('add_relu{}'.format(i), nn.LeakyReLU(0.1)))
        self.add_nn_stack = nn.Sequential(OrderedDict(add_nn))
        if depth - add_depth == 2 or tail_depth == 1:
            relu_nn = [('linear1', nn.Linear(r_bar * add_width, 1))]
        else:
            relu_nn = [('linear1', nn.Linear(r_bar * add_width, width)),
                       ('relu1', nn.LeakyReLU(0.1))]
            start_depth = math.ceil(len(self.add_nn_stack) / 2) + 2
            for i in range(start_depth, max(start_depth, depth)):
                relu_nn.append(('linear{}'.format(i), nn.Linear(width, width)))
                relu_nn.append(('relu{}'.format(i), nn.LeakyReLU(0.1)))
            relu_nn.append(('linear{}'.format(depth), nn.Linear(width, 1)))
        self.relu_stack = nn.Sequential(
            OrderedDict(relu_nn)
        )

    def forward(self, x, is_training=False, **kwargs):
        if self.use_input_dropout and is_training:
            x = self.input_dropout(x)
        x = self.add_nn_stack(x)
        x = self.relu_stack(x)
        x = x.reshape(-1, 1)
        return x


class RegressionNN(nn.Module):
    def __init__(self, p, depth, width, input_dropout=False, dropout_rate=0.0, oracle=False, check_depth=False, **kwargs):
        super(RegressionNN, self).__init__()
        self.use_input_dropout = input_dropout
        self.input_dropout = nn.Dropout(p=dropout_rate)
        self.batch_norm = nn.BatchNorm1d(p)
        relu_nn = [('linear1', nn.Linear(p, width)),
                   ('relu1', nn.LeakyReLU(0.1))]
        for i in range(2, depth + 1):
            relu_nn.append(('linear{}'.format(i), nn.Linear(width, width)))
            relu_nn.append(('relu{}'.format(i), nn.LeakyReLU(0.1)))
        relu_nn.append(('linear{}'.format(depth+1), nn.Linear(width, 1)))

        self.relu_stack = nn.Sequential(
            OrderedDict(relu_nn)
        )

    def forward(self, x, is_training=False, **kwargs):
        if self.use_input_dropout and is_training:
            x = self.input_dropout(x)
        pred = self.relu_stack(x)
        return pred


class Regression_bottleneck_NN(nn.Module):
    def __init__(self, p, depth, width, bottleneck_width, input_dropout=False, dropout_rate=0.0, check_depth=False, **kwargs):
        super().__init__()
        if check_depth:
            assert depth >= 3
        self.use_input_dropout = input_dropout
        self.input_dropout = nn.Dropout(p=dropout_rate)
        self.batch_norm = nn.BatchNorm1d(p)
        relu_nn = [('linear1', nn.Linear(p, bottleneck_width)),
                   ('relu1', nn.LeakyReLU(0.1))]
        relu_nn.extend(
            [('linear2', nn.Linear(bottleneck_width, width)), ('relu1', nn.LeakyReLU(0.1))])
        for i in range(3, depth):
            relu_nn.append(('linear{}'.format(i), nn.Linear(width, width)))
            relu_nn.append(('relu{}'.format(i), nn.LeakyReLU(0.1)))
        relu_nn.append(('linear{}'.format(depth), nn.Linear(width, 1)))

        self.relu_stack = nn.Sequential(
            OrderedDict(relu_nn)
        )

    def forward(self, x, is_training=False, **kwargs):
        if self.use_input_dropout and is_training:
            x = self.input_dropout(x)
        pred = self.relu_stack(x)
        return pred
