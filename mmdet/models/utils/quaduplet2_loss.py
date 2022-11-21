import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
        for i in range(len(targets)):
            if targets[i] < 0:
                bg.append(inputs[i])
            else:
                inputs_new.append(inputs[i])
                targets_new.append(targets[i])

        inputs_new = torch.stack(inputs_new)
        targets_new = torch.stack(targets_new)
        n = inputs_new.size(0)

        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs_new, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs_new, inputs_new.t())  # a^2 + b^2 - 2ab = (a - b)^2
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability, [P, P]

        # if self.use_hard_mining:
        #     uncertainty = torch.load("./saved_file/uncertainty.pth").cuda()
        #     dist = dist * uncertainty[index].unsqueeze(0)

        # For each anchor, find the hardest positive and negative
        mask = targets_new.expand(n, n).eq(targets_new.expand(n, n).t())
        dist_ap, dist_an = [], []
        p_idx, n_idx = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max(dim=0)[0])
            dist_an.append(dist[i][mask[i] == 0].min(dim=0)[0])
            p_idx.append(dist[i][mask[i]].max(dim=0)[1])
            n_idx.append(dist[i][mask[i] == 0].min(dim=0)[1])

        dist_ap = torch.stack(dist_ap)  # [B]
        dist_an = torch.stack(dist_an)  # [B]
        p_idx = torch.stack(p_idx)
        n_idx = torch.stack(n_idx)

        # if self.use_IoU_loss:
        #     clip_IoU = torch.clamp(IoU, self.IoU_loss_clip[0], self.IoU_loss_clip[1])    # [B, ]
        #     dist_ap = dist_ap * (clip_IoU * clip_IoU[p_idx])
        #     dist_an = dist_an * (clip_IoU * clip_IoU[n_idx])
        #     dist_ap = dist_ap * clip_IoU.detach()
        #     dist_an = dist_an * clip_IoU.detach()
            
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = torch.tensor(0.).to(y.device)
        loss_instance = self.ranking_loss(dist_an, dist_ap, y)
    
        # if self.use_hard_mining:
        #     uncertainty = torch.load("./saved_file/uncertainty.pth").cuda()
        #     loss_instance = loss_instance * uncertainty[index]
        
        loss_instance = loss_instance.mean()
        loss += self.instance_weight * loss_instance

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
            # For each anchor, find the hardest positive and negative
            # mask = targets_new.expand(n, ).eq(targets_new.expand(n, n).t())

            dist_ap, dist_an = [], []
            for i in range(n):
                dist_ap.append(dist[i].max())
                dist_an.append(dist_new[i].min())
            dist_ap = torch.stack(dist_ap)
            dist_an = torch.stack(dist_an)
            
            y = torch.ones_like(dist_an)
            loss_bg = self.ranking_loss(dist_an, dist_ap, y)
            
            # if self.use_hard_mining:
            #     uncertainty = torch.load("./saved_file/uncertainty.pth").cuda()
            #     loss_bg = loss_bg * uncertainty[index].detach()
            
            loss_bg = loss_bg.mean()
            loss += self.bg_weight * loss_bg
        return loss
