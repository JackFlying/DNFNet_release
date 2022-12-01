import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os 

class Quaduplet2Loss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    Args:
        margin (float): margin for triplet.
    """
    def __init__(self, margin=0.3, bg_weight=0.25, instance_weight=1, use_IoU_loss=False, IoU_loss_clip=[0.7, 1], use_uncertainty_loss=False,
                    use_hard_mining=False):
        super(Quaduplet2Loss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin, reduction='none')
        self.instance_weight = instance_weight
        self.bg_weight = bg_weight
        self.use_IoU_loss = use_IoU_loss
        self.use_uncertainty_loss = use_uncertainty_loss
        self.use_hard_mining = use_hard_mining
        self.IoU_loss_clip = IoU_loss_clip

    def forward(self, inputs, targets, index, IoU):
        """
        Does not calculate noise inputs with label -1
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (num_classes)
            index: memory index of person
        """
        inputs_new = []
        bg = []
        targets_new = []
        
        label_mask = torch.load(os.path.join('saved_file', 'label_mask.pth')).cuda()    # [N, ]
        target_label_mask = label_mask[index]
        
        for i in range(len(targets)):
            if targets[i] < 0:
                bg.append(inputs[i])
            else:
                inputs_new.append(inputs[i])
                targets_new.append(targets[i])

        inputs_new = torch.stack(inputs_new)
        targets_new = torch.stack(targets_new)
        # inputs_new = inputs_new[target_label_mask]
        # targets_new = targets_new[target_label_mask]
        n = inputs_new.size(0)
        loss = torch.tensor(0.).to(inputs_new.device)
        if n == 0:
            return loss
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs_new, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs_new, inputs_new.t())  # a^2 + b^2 - 2ab = (a - b)^2
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability, [P, P]

        # For each anchor, find the hardest positive and negative
        mask = targets_new.expand(n, n).eq(targets_new.expand(n, n).t())
        dist_ap, dist_an = [], []
        # p_idx, n_idx = [], []
        
        # 需要同时满足具有正样本和负样本
        try:
            for i in range(n):
                dist_ap.append(dist[i][mask[i]].max(dim=0)[0])
                dist_an.append(dist[i][mask[i] == 0].min(dim=0)[0])
                # p_idx.append(dist[i][mask[i]].max(dim=0)[1])
                # n_idx.append(dist[i][mask[i] == 0].min(dim=0)[1])

            dist_ap = torch.stack(dist_ap)  # [B]
            dist_an = torch.stack(dist_an)  # [B]
            # p_idx = torch.stack(p_idx)
            # n_idx = torch.stack(n_idx)

            # Compute ranking hinge loss
            y = torch.ones_like(dist_an)
            loss_instance = self.ranking_loss(dist_an, dist_ap, y)
            # loss_instance = loss_instance * target_label_mask
            loss_instance = loss_instance.mean()
            loss += self.instance_weight * loss_instance
        except:
            pass

        try:
            bg = torch.stack(bg)
            m = bg.size(0)
        except:
            m = 0
            
        if m > 0:
            dist_p = torch.pow(inputs_new, 2).sum(dim=1, keepdim=True).expand(n, m)
            dist_bg = torch.pow(bg, 2).sum(dim=1, keepdim=True)
            dist_bg = dist_bg.expand(m, n)
            dist_new = dist_p + dist_bg.t()
            dist_new.addmm_(1, -2, inputs_new, bg.t())
            dist_new = dist_new.clamp(min=1e-12).sqrt()  # for numerical stability

            dist_ap, dist_an = [], []
            try:
                for i in range(n):
                    dist_ap.append(dist[i].max())
                    dist_an.append(dist_new[i].min())
                dist_ap = torch.stack(dist_ap)
                dist_an = torch.stack(dist_an)
                
                y = torch.ones_like(dist_an)
                loss_bg = self.ranking_loss(dist_an, dist_ap, y)
                # loss_bg = loss_bg * target_label_mask
                loss_bg = loss_bg.mean()
                loss += self.bg_weight * loss_bg
            except:
                pass
        return loss
