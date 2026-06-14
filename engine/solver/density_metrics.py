import torch
import math

class DensityCountRMSE:
    def __init__(self):
        self.squared_error_sum = 0.0
        self.num_images = 0

    @torch.no_grad()
    def update(self, pred_densitymap, targets):

        pred_counts = pred_densitymap.sum(
            dim=(1,2,3)
        ).cpu()

        gt_counts = torch.tensor(
            [len(t["labels"]) for t in targets],
            dtype=torch.float32
        )

        error = pred_counts - gt_counts

        self.squared_error_sum += (error ** 2).sum().item()

        self.num_images += len(targets)

    def compute(self):

        return math.sqrt(
            self.squared_error_sum /
            max(self.num_images,1)
        )

# density_metrics.py

import torch


class DensityKL:
    def __init__(self):
        self.kl_sum = 0.0
        self.num_images = 0

    @torch.no_grad()
    def update(self, pred_densitymap, gt_densitymap):

        pred = pred_densitymap.flatten()

        gt = gt_densitymap.flatten()

        pred = pred / (pred.sum() + 1e-8)
        gt = gt / (gt.sum() + 1e-8)

        kl = torch.sum(
            gt * torch.log(
                (gt + 1e-8) /
                (pred + 1e-8)
            )
        )

        self.kl_sum += kl.item()
        self.num_images += 1

    def compute(self):

        return self.kl_sum / max(self.num_images, 1)

class DensityNSS:
    def __init__(self):

        self.nss_sum = 0.0
        self.num_images = 0

    @torch.no_grad()
    def update(self, pred_densitymap, gt_densitymap):

        pred = pred_densitymap.squeeze()

        pred_norm = (
            pred - pred.mean()
        ) / (
            pred.std() + 1e-8
        )

        fixation = gt_densitymap > 0

        if fixation.sum() > 0:

            nss = pred_norm[
                fixation
            ].mean()

            self.nss_sum += nss.item()

            self.num_images += 1

    def compute(self):

        return self.nss_sum / max(self.num_images, 1)

# class DensityDDS:
#     def __init__(self):
#         self.kl_sum = 0.0
#         self.num_images = 0

#     @torch.no_grad()
#     def update(self, pred_density, gt_density):

#         eps = 1e-8

#         pred_density = pred_density.float()
#         gt_density = gt_density.float()

#         pred_prob = pred_density / (pred_density.sum() + eps)
#         gt_prob = gt_density / (gt_density.sum() + eps)

#         kl = (
#             gt_prob *
#             torch.log(
#                 (gt_prob + eps) /
#                 (pred_prob + eps)
#             )
#         ).sum()

#         self.kl_sum += kl.item()
#         self.num_images += 1

#     def compute(self):

        mean_kl = self.kl_sum / max(self.num_images, 1)

        dds = 1.0 / (1.0 + mean_kl)

        return dds