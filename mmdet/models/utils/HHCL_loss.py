import collections
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, autograd


class CrossEntropyLabelSmooth(nn.Module):
	"""Cross entropy loss with label smoothing regularizer.
	Reference:
	Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
	Equation: y = (1 - epsilon) * y + epsilon / K.
	Args:
		num_classes (int): number of classes.
		epsilon (float): weight.
	"""

	def __init__(self, num_classes=0, epsilon=0.1, topk_smoothing=False):
		super(CrossEntropyLabelSmooth, self).__init__()
		self.num_classes = num_classes
		self.epsilon = epsilon
		self.logsoftmax = nn.LogSoftmax(dim=1).cuda()
		self.k = 1 if not topk_smoothing else self.num_classes//50

	def forward(self, inputs, targets):
		"""
		Args:
			inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
			targets: ground truth labels with shape (num_classes)
		"""
		log_probs = self.logsoftmax(inputs)
		if self.k >1:
			topk = torch.argsort(-log_probs)[:,:self.k]
			targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1 - self.epsilon)
			targets += torch.zeros_like(log_probs).scatter_(1, topk, self.epsilon / self.k)
		else:
			targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
			targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
		loss = (- targets * log_probs).mean(0).sum()
		return loss

class CM(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update
        for x, y in zip(inputs, targets):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None


def cm(inputs, indexes, features, momentum=0.5):
    return CM.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class CM_Hard(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        batch_centers = collections.defaultdict(list)
        for instance_feature, index in zip(inputs, targets.tolist()):
            batch_centers[index].append(instance_feature)

        for index, features in batch_centers.items():
            distances = []
            for feature in features:
                distance = feature.unsqueeze(0).mm(ctx.features[index].unsqueeze(0).t())[0][0]
                distances.append(distance.cpu().numpy())

            median = np.argmin(np.array(distances))
            ctx.features[index] = ctx.features[index] * ctx.momentum + (1 - ctx.momentum) * features[median]
            ctx.features[index] /= ctx.features[index].norm()

        return grad_inputs, None, None, None


def cm_hard(inputs, indexes, features, momentum=0.5):
    return CM_Hard.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class CM_Hybrid(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        nums = len(ctx.features)//2
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        batch_centers = collections.defaultdict(list)
        for instance_feature, index in zip(inputs, targets.tolist()):   # 将输入特征按照伪标签进行分类
            batch_centers[index].append(instance_feature)

        # calculate the distance between inputs and features in the memory
        for index, features in batch_centers.items():
            distances = []
            for feature in features:
                distance = feature.unsqueeze(0).mm(ctx.features[index].unsqueeze(0).t())[0][0]
                distances.append(distance.cpu().numpy())
            # update hard memory by the most similar instance in mini batch
            median = np.argmin(np.array(distances))
            ctx.features[index] = ctx.features[index] * ctx.momentum + (1 - ctx.momentum) * features[median]
            ctx.features[index] /= ctx.features[index].norm()
            # update cluster memory by mean mini batch
            mean = torch.stack(features, dim=0).mean(0)
            ctx.features[index+nums] = ctx.features[index+nums] * ctx.momentum + (1 - ctx.momentum) * mean
            ctx.features[index+nums] /= ctx.features[index+nums].norm()

        return grad_inputs, None, None, None


def cm_hybrid(inputs, indexes, features, momentum=0.5):
    return CM_Hybrid.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class CM_Hybrid_v2(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum, num_instances):
        ctx.features = features
        ctx.momentum = momentum
        ctx.num_instances = num_instances
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        nums = len(ctx.features)//(ctx.num_instances + 1)
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        batch_centers = collections.defaultdict(list)
        updated = set()
        for k, (instance_feature, index) in enumerate(zip(inputs, targets.tolist())):
            batch_centers[index].append(instance_feature)
            if index not in updated:
                indexes = [index + nums*i for i in range(1, (targets==index).sum()+1)]
                ctx.features[indexes] = inputs[targets==index]
                # ctx.features[indexes] = ctx.features[indexes] * ctx.momentum + (1 - ctx.momentum) * inputs[targets==index]
                # ctx.features[indexes] /= ctx.features[indexes].norm(dim=1, keepdim=True)
                updated.add(index)

        for index, features in batch_centers.items():
            mean = torch.stack(features, dim=0).mean(0)
            ctx.features[index] = ctx.features[index] * ctx.momentum + (1 - ctx.momentum) * mean
            ctx.features[index] /= ctx.features[index].norm()
        
        return grad_inputs, None, None, None, None


def cm_hybrid_v2(inputs, indexes, features, momentum=0.5, num_instances=16, *args):
    return CM_Hybrid_v2.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device), num_instances)


class ClusterMemory(nn.Module):
    
    __CMfactory = {
        'CM': cm,
        'CMhard':cm_hard,
    }

    def __init__(self, num_features, num_samples=0, temp=0.05, momentum=0.2, mode='CM', hard_weight=0.5, smooth=0., num_instances=1, num_memory=18048):
        super(ClusterMemory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples  # num_cluster
        self.num_memory = num_memory

        self.momentum = momentum
        self.temp = temp
        self.cm_type = mode
        self.idx = torch.zeros(num_memory).long()

        if smooth > 0:
            self.cross_entropy = CrossEntropyLabelSmooth(self.num_samples, 0.1, True)
            print('>>> Using CrossEntropy with Label Smoothing.')
        else: 
            self.cross_entropy = nn.CrossEntropyLoss().cuda() 

        if self.cm_type in ['CM', 'CMhard']:
            self.register_buffer('features', torch.zeros(num_samples, num_features))
        elif self.cm_type=='CMhybrid':
            self.hard_weight = hard_weight
            print('hard_weight: {}'.format(self.hard_weight))
            self.register_buffer('instance_features', torch.zeros(num_memory, num_features))
            self.register_buffer("labels", torch.zeros(num_memory).long())
        elif self.cm_type=='CMhybrid_v2':
            self.hard_weight = hard_weight
            self.num_instances = num_instances
            self.register_buffer('features', torch.zeros((self.num_instances + 1) * num_samples, num_features))
        else:
            raise TypeError('Cluster Memory {} is invalid!'.format(self.cm_type))

    @torch.no_grad()
    def register_features_buffer(self, num_samples):
        self.register_buffer("features", torch.zeros(2 * num_samples, self.num_features))
        self.features = self.features.to(self.labels.device)

    @torch.no_grad()
    def _init_ids(self, ids):
        self.idx.data.copy_(ids.long().to(self.labels.device))

    @torch.no_grad()
    def get_cluster_ids(self, indexes):
        return self.labels[indexes].clone()

    @torch.no_grad()
    def _update_feature(self, features):
        features = F.normalize(features, p=2, dim=1)
        self.features.data.copy_(features.float().to(self.labels.device))

    @torch.no_grad()
    def _update_instance_feature(self, features):
        features = F.normalize(features, p=2, dim=1)
        self.instance_features.data.copy_(features.float().to(self.labels.device))

    @torch.no_grad()
    def _update_label(self, labels):
        self.labels.data.copy_(labels.long().to(self.labels.device))

    def forward(self, inputs, targets):
        # target is the Pseudo labels
        if self.cm_type in ['CM', 'CMhard']:
            outputs = ClusterMemory.__CMfactory[self.cm_type](inputs, targets, self.features, self.momentum)
            outputs /= self.temp
            loss = self.cross_entropy(outputs, targets)
            return loss

        elif self.cm_type=='CMhybrid':
            outputs = cm_hybrid(inputs, targets, self.features, self.momentum)
            outputs /= self.temp
            output_hard, output_mean = torch.chunk(outputs, 2, dim=1)
            loss = self.hard_weight * (self.cross_entropy(output_hard, targets) + (1 - self.hard_weight) * self.cross_entropy(output_mean, targets))
            return loss

        elif self.cm_type=='CMhybrid_v2':
            outputs = cm_hybrid_v2(inputs, targets, self.features, self.momentum, self.num_instances)
            out_list = torch.chunk(outputs, self.num_instances + 1, dim=1)
            out = torch.stack(out_list[1:], dim=0)
            neg = torch.max(out, dim=0)[0]
            pos = torch.min(out, dim=0)[0]
            mask = torch.zeros_like(out_list[0]).scatter_(1, targets.unsqueeze(1), 1)
            logits = mask * pos + (1-mask) * neg
            loss = self.hard_weight * self.cross_entropy(out_list[0] / self.temp, targets) \
                + (1 - self.hard_weight) * self.cross_entropy(logits / self.temp, targets)
            return loss
