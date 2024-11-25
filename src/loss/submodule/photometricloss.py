"""
    CompletionFormer
    ======================================================================
    Photometric loss implementation

"""


import torch
import torch.nn as nn

from . import criteria
from .inverse_warp import Intrinsics, homography_from



class PhotometricLoss(nn.Module):
    def __init__(self, args):
        super(PhotometricLoss, self).__init__()

        self.args = args
        self.photometric_criterion = criteria.PhotometricLoss()
        self.avgpool = torch.nn.AvgPool2d(kernel_size=2, stride=2).cuda()

    def multiscale(self, img):
        img1 = self.avgpool(img)
        img2 = self.avgpool(img1)
        img3 = self.avgpool(img2)
        img4 = self.avgpool(img3)
        img5 = self.avgpool(img4)
        return img5, img4, img3, img2, img1
        # return img4, img3, img2, img1

    def forward(self, sample, output):
        mask = (output['pred'] < 1e-3).float()
        photometric_loss = 0

        # Loss 2: the self-supervised photometric loss
        if self.args.use_pose:
            # create multi-scale pyramids
            pred_array = self.multiscale(output['pred'])
            rgb_curr_array = self.multiscale(sample['rgb'])
            rgb_near_array = self.multiscale(sample['rgb_near'])

            if mask is not None:
                mask_array = self.multiscale(mask)
            num_scales = len(pred_array)

            fu, fv = sample['K'][:, :1], sample['K'][:, 1:2]
            cu, cv = sample['K'][:, 2:3], sample['K'][:, 3:4]

            kitti_intrinsics = Intrinsics(self.args.width, self.args.height, fu, fv, cu, cv)

            if torch.cuda.is_available():
                kitti_intrinsics = kitti_intrinsics.cuda()

            for scale in range(len(pred_array)):
                # compute photometric loss at multiple scales
                pred_ = pred_array[scale]
                rgb_curr_ = rgb_curr_array[scale]
                rgb_near_ = rgb_near_array[scale]
                mask_ = None
                if mask is not None:
                    mask_ = mask_array[scale]

                # compute the corresponding intrinsic parameters
                height_, width_ = pred_.size(2), pred_.size(3)
                intrinsics_ = kitti_intrinsics.scale(height_, width_)

                # inverse warp from a nearby frame to the current frame
                r_mat = torch.tensor(sample['r_mat'], dtype=torch.float32).cuda()
                t_vec = torch.tensor(sample['t_vec'], dtype=torch.float32).cuda()

                warped_ = homography_from(rgb_near_, pred_, r_mat, t_vec, intrinsics_)

                photometric_loss += self.photometric_criterion(rgb_curr_, warped_, mask_) * (2 ** (scale - num_scales))

            if isinstance(photometric_loss, torch.Tensor):
                photometric_loss = photometric_loss.clone().detach().requires_grad_(True)
            else:
                photometric_loss = torch.tensor(photometric_loss, requires_grad=True)

        return photometric_loss
