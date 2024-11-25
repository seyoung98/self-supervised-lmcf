"""
    CompletionFormer
    ======================================================================

    Smoothness loss implementation
"""


import torch
import torch.nn as nn

from . import criteria


class SmoothnessLoss(nn.Module):
    def __init__(self, args):
        super(SmoothnessLoss, self).__init__()

        self.args = args
        self.smoothness_criterion = criteria.SmoothnessLoss()

    def forward(self, sample, output):
        smoothness_loss = self.smoothness_criterion(output['pred'])

        return smoothness_loss
