import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os 

def euclidean_dist(x, y):
	m, n = x.size(0), y.size(0)
	xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
	yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
	dist = xx + yy
	dist.addmm_(1, -2, x, y.t())
	dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
	return dist

def cosine_dist(x, y):
	bs1, bs2 = x.size(0), y.size(0)
	frac_up = torch.matmul(x, y.transpose(0, 1))
	frac_down = (torch.sqrt(torch.sum(torch.pow(x, 2), 1))).view(bs1, 1).repeat(1, bs2) * \
	            (torch.sqrt(torch.sum(torch.pow(y, 2), 1))).view(1, bs2).repeat(bs1, 1)
	cosine = frac_up / frac_down
	return 1-cosine

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

    def forward(self, inputs, targets, index=None, IoU=None):
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
        # indexs_new = []
        inputs = F.normalize(inputs)
        # inputs /= inputs.norm()
        for i in range(len(targets)):
            if targets[i] < 0:
                bg.append(inputs[i])
            else:
                inputs_new.append(inputs[i])
                targets_new.append(targets[i])
                # indexs_new.append(index[i])

        inputs_new = torch.stack(inputs_new)
        targets_new = torch.stack(targets_new)
        # indexs_new = torch.stack(indexs_new)
        
        # label_mask = torch.load(os.path.join('saved_file', 'label_mask.pth')).cuda()    # [N, ]
        # target_label_mask = label_mask[indexs_new]
        # inputs_new = inputs_new[target_label_mask]
        # targets_new = targets_new[target_label_mask]
        
        n = inputs_new.size(0)
        loss = torch.tensor(0.).to(inputs_new.device)
        if n == 0:
            return loss
        # Compute pairwise distance, replace by the official when merged
        dist = euclidean_dist(inputs_new, inputs_new)
        # dist = cosine_dist(inputs_new, inputs_new)
  
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
            dist_new = euclidean_dist(inputs_new, bg)
            # dist_new = cosine_dist(inputs_new, bg)
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
