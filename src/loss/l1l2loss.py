"""
    Non-Local Spatial Propagation Network for Depth Completion
    Jinsun Park, Kyungdon Joo, Zhe Hu, Chi-Kuei Liu and In So Kweon

    European Conference on Computer Vision (ECCV), Aug 2020

    Project Page : https://github.com/zzangjinsun/NLSPN_ECCV20
    Author : Jinsun Park (zzangjinsun@kaist.ac.kr)

    ======================================================================

    NLSPNLoss implementation
"""

from . import BaseLoss
import torch


class L1L2Loss(BaseLoss):
    def __init__(self, args):
        super(L1L2Loss, self).__init__(args)

        self.loss_name = []
        self.t_valid = 0.0001

        for k, _ in self.loss_dict.items():
            self.loss_name.append(k)

    def compute(self, sample, output):
        loss_val = []

        for idx, loss_type in enumerate(self.loss_dict):
            loss = self.loss_dict[loss_type]
            loss_func = loss['func']
            if loss_func is None:
                continue


            loss_tmp = 0.0
            if loss_type in ['Depth', 'Photometric', 'Smoothness']:
                loss_tmp += loss_func(sample, output) * 1.0
            else:
                raise NotImplementedError

            loss_tmp = loss['weight'] * loss_tmp
            loss_tmp = loss_tmp.to(sample['rgb'].device)
            loss_val.append(loss_tmp)

        loss_val = torch.stack(loss_val)

        loss_sum = torch.sum(loss_val, dim=0, keepdim=True)

        loss_val = torch.cat((loss_val, loss_sum))
        loss_val = torch.unsqueeze(loss_val, dim=0).detach()

        return loss_sum, loss_val
