"""
    CompletionFormer
    ======================================================================

    Depth loss implementation
"""


import torch
import torch.nn as nn

from . import criteria


class DepthLoss(nn.Module):
    def __init__(self, args):
        super(DepthLoss, self).__init__()

        self.args = args
        self.depth_criterion = criteria.MaskedMSELoss() if (
                args.criterion == 'l2') else criteria.MaskedL1Loss()

    def forward(self, sample, output):
        # Loss 1: the direct depth supervision from ground truth label
        # mask=1 indicates that a pixel does not ground truth labels
        depth_loss = self.depth_criterion(output['pred'], sample['dep'])

        return depth_loss
