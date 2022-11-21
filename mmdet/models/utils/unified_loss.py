from typing import Tuple
import torch
from torch import nn, Tensor

def convert_label_to_similarity(normed_feature: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
    similarity_matrix = normed_feature @ normed_feature.transpose(1, 0)
    label_matrix = label.unsqueeze(1) == label.unsqueeze(0)

    positive_matrix = label_matrix.triu(diagonal=1)
    negative_matrix = label_matrix.logical_not().triu(diagonal=1)

    similarity_matrix = similarity_matrix.view(-1)
    positive_matrix = positive_matrix.view(-1)
    negative_matrix = negative_matrix.view(-1)
    return similarity_matrix[positive_matrix], similarity_matrix[negative_matrix]

class UnifiedLoss(nn.Module):
    def __init__(self, gamma: float) -> None:
        super(UnifiedLoss, self).__init__()
        self.gamma = gamma
        self.soft_plus = nn.Softplus()  # log(1 + exp(x))

    def forward(self, normed_feature: Tensor, label: Tensor) -> Tensor:
        sp, sn = convert_label_to_similarity(normed_feature, label)
        logit_p = - sp * self.gamma
        logit_n = sn * self.gamma

        loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))
    
        return loss
