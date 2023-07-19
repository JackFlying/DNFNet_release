import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import collections
import os

def instance_label(labels):
    labels = labels[0]
    sample_num = len(labels)
    new_labels = np.arange(sample_num)
    return [new_labels]

def get_outlier(labels):
    """
        将outlier的标签设置为-1
        # 其它设置为连续的标签
    """
    labels = labels[0]
    labels_num = collections.defaultdict(int)
    for label in labels:
        labels_num[label] += 1
    # sample_outlier
    sample_outlier = torch.zeros(len(labels)).bool()
    for i, label in enumerate(labels):
        if labels_num[label] == 1:
            sample_outlier[i] = True
        else:
            sample_outlier[i] = False
    new_labels = torch.tensor(labels).long()
    # 如果把异常点的label设置为-1, 不容易update
    new_labels[sample_outlier == False] = torch.tensor(reassignment_labels(new_labels[sample_outlier == False].tolist())).long()
    new_labels[sample_outlier == True] = torch.tensor(reassignment_labels(new_labels[sample_outlier == True].tolist())).long()
    new_labels[sample_outlier == True] += new_labels[sample_outlier == False].max() + 1 # 异常点编号在labeled之后

    # cluster outlier
    labels_num = collections.defaultdict(int)
    for label in new_labels:
        labels_num[label.item()] += 1
    
    cluster_outlier = torch.zeros(new_labels.max() + 1).bool()
    for lb, lb_num in labels_num.items():
        if(lb_num == 1):
            cluster_outlier[lb] = True
        else:
            cluster_outlier[lb] = False

    torch.save(sample_outlier, os.path.join("saved_file", "sample_outlier.pth"))
    torch.save(cluster_outlier, os.path.join("saved_file", "cluster_outlier.pth"))
    new_labels = new_labels.tolist()
    return [new_labels]

def get_weighted_mean(memory_features): # 只能求一个簇
    """
        距离聚类中心越近，权重越高
        memory_features: [K, D]
    """
    mean = torch.mean(memory_features, dim=0)
    return mean
    cos_sim = 1 - memory_features.mm(mean[None].t())    # 余弦相似度越小,越相似
    cos_sim_sf = F.softmax(cos_sim / 0.1, dim=0)
    weighted_mean = torch.sum(memory_features * cos_sim_sf, dim=0)
    return weighted_mean

def get_uncertainty_by_centroid(labels, features, logger, T):
    # old_labels = labels
    labels = torch.tensor(labels[0])
    features = torch.tensor(features[0])
    new_labels = labels
    last_uncertainty_num = 0
    for t in range(T):
        labels = new_labels
        unique_labels = torch.unique(labels)
        centroid = []
        for ul in unique_labels:
            mean = get_weighted_mean(features[labels == ul])
            centroid.append(mean)
        centroid = torch.stack(centroid)
        sim = features.mm(centroid.t())
        argmax = sim.argmax(dim=-1)
        # 与自身prototype的距离为d1，与最近非自身prototype的距离是d2，计算ratio=d1/d2。如果ratio>=1,则说明不是异常点;如果ratio<1,则是异常点,则ratio作为损失的权重.
        # d1 = sim[torch.arange(sim.shape[0]), labels]
        # d2 = sim[torch.arange(sim.shape[0]), argmax]
        # ratio = d1 / (d2 + 1e-6)
        # torch.save(ratio, os.path.join("saved_file", "ratio.pth"))
        uncertainty = (argmax == labels)
        new_labels = transfer_label_noise_to_outlier(uncertainty, argmax.tolist())[0]
        new_labels = torch.tensor(new_labels)
        logger.info(f"{t}: uncertainty sample number: " + str(len(uncertainty[uncertainty == False])))
        logger.info(f"{t}: certainty sample number: " + str(len(uncertainty[uncertainty == True])))
        if last_uncertainty_num == len(uncertainty[uncertainty == False]):
            break
        else:
            last_uncertainty_num = len(uncertainty[uncertainty == False])
    # torch.save(uncertainty, os.path.join("saved_file", "uncertainty.pth"))
    
    return new_labels

def get_weight_by_uncertainty(uncertainty, labels):
    labels = torch.tensor(labels)
    unique_labels = torch.unique(labels)
    for ul in unique_labels:
        uncertainty[labels == ul] = F.softmax(uncertainty[labels == ul], dim=-1)
    return uncertainty

def get_mean_conv(input_vec):
    mean = torch.mean(input_vec, axis=0)
    x = input_vec - mean
    cov_matrix = torch.matmul(x.T, x) / (x.shape[0] - 1 if x.shape[0] > 1 else 1)
    return mean, cov_matrix 

def get_gaussion_distributation(memory_features, memory_labels):
    unique_labels = torch.unique(memory_labels)
    means, stds = [], []
    for ul in unique_labels:
        mean, std = get_mean_conv(memory_features[memory_labels == ul])
        means.append(mean)
        stds.append(std)
    means = torch.stack(means, dim=0)
    stds = torch.stack(stds, dim=0)
    return means, stds

def GMM(memory_features, memory_features_std, memory_labels):
    unique_labels = torch.unique(memory_labels)
    memory_features_std = torch.exp(memory_features_std)
    means, stds = [], []
    for ul in unique_labels:
        gmm_mean = torch.mean(memory_features[memory_labels == ul], dim=0)
        gmm_var = torch.mean(memory_features_std[memory_labels == ul], dim=0)
        means.append(gmm_mean)
        stds.append(gmm_var)
    means = torch.stack(means, dim=0)
    stds = torch.stack(stds, dim=0)
    return means, stds

def get_softlabel_by_part(pseudo_labels, pseudo_label2s, memory_features):

    pseudo_labels = torch.tensor(pseudo_labels) # [1, 500]
    pseudo_label2s = torch.tensor(pseudo_label2s)   #[1, 500]
    iou_mat = compute_label_iou_matrix(pseudo_label2s, pseudo_labels)
    norm_iou_mat = (iou_mat.t() / iou_mat.t().sum(0)).t()   # [pseudo_label2s_numbers, pseudo_labels_numbers]
    # uncertainty = norm_iou_mat.t()[pseudo_labels[0]][torch.arange(len(pseudo_labels[0])), pseudo_label2s[0]]

    pred_temp = 30
    label_center2s = generate_cluster_features(pseudo_label2s[0].tolist(), memory_features[0])
    probs_perv = extract_probabilities(memory_features[0], label_center2s, pred_temp)
    N, C = probs_perv.size(0), probs_perv.size(1)
    onehot_labels = torch.full(size=(N, C), fill_value=0)   # [18048, 6104]
    onehot_labels.scatter_(dim=1, index=torch.unsqueeze(pseudo_label2s[0], dim=1), value=1)
    alpha = 0.9
    probs_perv = alpha * onehot_labels + (1.0 - alpha) * probs_perv # pseudo_label2s修正pseudo_labels

    prob_soft_labels = probs_perv.mm(norm_iou_mat)
    beta = 0
    hard_iou_labels = compute_sample_softlabels(pseudo_label2s, pseudo_labels, "iou", "original")
    sample_soft_labels = beta * hard_iou_labels + (1.0 - beta) * prob_soft_labels   # 软硬标签的综合程度, [N, pseudo_labels_numbers]
    # uncertainty = sample_soft_labels[torch.arange(len(pseudo_labels[0])), pseudo_labels[0]]
    return sample_soft_labels

def get_uncertainty_by_part(pseudo_labels, pseudo_label2s, memory_features, logger, cfg):
    print("----------------------------get_uncertainty_by_part----------------------------")
    pseudo_labels = torch.tensor(pseudo_labels) # [1, 500]
    pseudo_label2s = torch.tensor(pseudo_label2s)   #[1, 500]
    iou_mat = compute_label_iou_matrix(pseudo_label2s, pseudo_labels)
    norm_iou_mat = (iou_mat.t() / iou_mat.t().sum(0)).t()   # [pseudo_label2s_numbers, pseudo_labels_numbers]
    uncertainty = norm_iou_mat.t()[pseudo_labels[0]][torch.arange(len(pseudo_labels[0])), pseudo_label2s[0]]
    # torch.save(uncertainty, os.path.join("saved_file", "uncertainty.pth"))
    # pred_temp = 30
    # label_center2s = generate_cluster_features(pseudo_label2s[0].tolist(), memory_features[0])
    # probs_perv = extract_probabilities(memory_features[0], label_center2s, pred_temp)
    # N, C = probs_perv.size(0), probs_perv.size(1)
    # onehot_labels = torch.full(size=(N, C), fill_value=0)   # [18048, 6104]
    # onehot_labels.scatter_(dim=1, index=torch.unsqueeze(pseudo_label2s[0], dim=1), value=1)
    # alpha = 0.9
    # probs_perv = alpha * onehot_labels + (1.0 - alpha) * probs_perv # pseudo_label2s修正pseudo_labels

    # prob_soft_labels = probs_perv.mm(norm_iou_mat)
    # beta = 0
    # hard_iou_labels = compute_sample_softlabels(pseudo_label2s, pseudo_labels, "iou", "original")
    # sample_soft_labels = beta * hard_iou_labels + (1.0 - beta) * prob_soft_labels   # 软硬标签的综合程度, [N, pseudo_labels_numbers]
    # uncertainty = sample_soft_labels[torch.arange(len(pseudo_labels[0])), pseudo_labels[0]]
    uncertainty_threshold = cfg.PSEUDO_LABELS.part_feat.uncertainty_threshold
    logger.info("uncertainty > uncertainty_threshold: " + str(len(uncertainty[uncertainty > uncertainty_threshold])))
    logger.info("uncertainty < uncertainty_threshold: " + str(len(uncertainty[uncertainty <= uncertainty_threshold])))
    uncertainty[uncertainty > uncertainty_threshold] = 1
    uncertainty[uncertainty <= uncertainty_threshold] = 0
    return uncertainty


def transfer_outlier_label(labels):
    """
        将outlier的标签转换为-1
    """
    # import ipdb;    ipdb.set_trace()
    counts = collections.defaultdict(int)
    for lb in labels:
        counts[lb] += 1
    for i in range(len(labels)):
        if counts[labels[i]] == 1:
            labels[i] = -1
    return labels

def RefineClusterProcess(Reference_Cluster_result, Target_Cluster_result, divide_ratio):
    # 异常点标签转换为-1
    Reference_Cluster_result = transfer_outlier_label(Reference_Cluster_result)
    Target_Cluster_result = transfer_outlier_label(Target_Cluster_result)
    
    Reference_Cluster_result = np.array(Reference_Cluster_result)
    Target_Cluster_result = np.array(Target_Cluster_result)
    
    L = len(Reference_Cluster_result)
    assert L == len(Target_Cluster_result)

    Target_Cluster_nums = len(set(Target_Cluster_result)) - (1 if -1 in Target_Cluster_result else 0)

    Final_Cluster = np.zeros(L, dtype=np.int64) - 1
    assert len(np.where(Final_Cluster == -1)[0]) == L

    # import ipdb;    ipdb.set_trace()
    ban_cluster = 0
    # -1指的是异常点
    for Target_Cluster in range(Target_Cluster_nums):
        Target_Cluster_index = np.where(Target_Cluster_result == Target_Cluster)[0]

        zero_index = np.where(Reference_Cluster_result == -1)[0]
        Target_Cluster_index = np.setdiff1d(Target_Cluster_index, zero_index)   # 返回在Target_Cluster_index中而不再zero_index中的数值

        if np.size(Target_Cluster_index) == 0:
            ban_cluster += 1
            continue
        num_ID = len(Target_Cluster_index)
        num_Part = np.bincount(Reference_Cluster_result[Target_Cluster_index])  # 统计从0到array数组中最大数字出现的个数的函数
        ban_flag = True

        for i in range(int(1/divide_ratio)):
            _max = np.argmax(num_Part)

            if num_Part[_max] > 0 and num_Part[_max] > num_ID * divide_ratio:
                Reference_Cluster_index = np.where(Reference_Cluster_result == _max)[0]
                fit_condition = np.intersect1d(Target_Cluster_index, Reference_Cluster_index)
                Final_Cluster[fit_condition] = Target_Cluster - ban_cluster
                num_Part[_max] = 0
                ban_flag = False
            else:
                break
        if ban_flag:
            ban_cluster += 1
    # import ipdb;    ipdb.set_trace()
    # 异常点分配数字
    label_mask = Final_Cluster != -1
    cluster_nums = len(set(Final_Cluster)) - (1 if -1 in Final_Cluster else 0)
    for i in range(len(Final_Cluster)):
        if(Final_Cluster[i] == -1):
            Final_Cluster[i] = cluster_nums
            cluster_nums += 1
    return [Final_Cluster]

def give_unknown_id():
    gt_person_ids = torch.load(os.path.join("saved_file", "person_ids.pth"))
    max_ids = gt_person_ids.max()
    max_ids += 1
    for i, label in enumerate(gt_person_ids):
        if label.item() == -1:
            gt_person_ids[i] = max_ids
        max_ids +=1 
    return gt_person_ids.tolist()

def get_is_known_list(pseudo_label):
    """
        pseudo_label: list
    """
    label_num = collections.defaultdict(list)
    for i, label in enumerate(pseudo_label):
        label_num[label].append(i)    
    is_known = torch.ones((len(pseudo_label))).bool()
    for i, label in enumerate(pseudo_label):
        is_known[i] = len(label_num[label]) > 1
    torch.save(is_known, os.path.join("saved_file", "is_known.pth"))

@torch.no_grad()
def transfer_label_noise_to_outlier(uncertaintys, labels):
    """
        uncertainty: [N]
        labels: [N]
    """
    print("transfer noisy label to outlier")
    mask = torch.ones(len(labels)).bool()
    # 计算簇的大小
    labels = torch.tensor(labels)
    num_classes = labels.max() + 1
    uncertaintys = uncertaintys.tolist()
    print("old num of classes", num_classes)
    new_labels = labels.clone()
    # 为hard sample赋予伪标签
    for i, (uncertainty, label) in enumerate(zip(uncertaintys, labels)):
        if uncertainty == 0:
            new_labels[i] = num_classes
            num_classes += 1
            mask[i] = False
    new_labels = reassignment_labels(new_labels.tolist())
    print("new num of classes", max(new_labels) + 1)
    torch.save(mask, os.path.join("saved_file", "mask.pth"))
    return [new_labels]
    
    # hard sample伪标签保持为-1
    # for i, (uncertainty, label) in enumerate(zip(uncertaintys, labels)):
    #     if uncertainty == 0:
    #         new_labels[i] = -1
    #         mask[i] = False
    # re_new_labels = reassignment_labels2(new_labels.tolist())
    # print("new num of classes", max(re_new_labels) + 1)
    # return [re_new_labels], mask

@torch.no_grad()
def reassignment_labels2(pseudo_label):
    """
        给标签赋予新的顺序,解决中间断层的问题
        -1不赋予标签
    """
    # 记录用到的标签
    old_labels_set = collections.defaultdict(list)
    for i, label in enumerate(pseudo_label):
        old_labels_set[label].append(i)
    new_labels_set = collections.defaultdict(list)
    # 赋予新的伪标签
    cnt = 1
    for new_label, old_label in enumerate(old_labels_set.keys()):
        if old_label == -1:
            new_labels_set[old_label] = 0   # 异常点标签赋值为0
        else:
            new_labels_set[old_label] = cnt   # 将旧标签映射到新的标签,保证中间没有间断
            cnt += 1

    new_pseudo_label = []
    for label in pseudo_label:
        new_pseudo_label.append(new_labels_set[label])
    
    return new_pseudo_label

@torch.no_grad()
def reassignment_labels(pseudo_label):
    """
        给标签赋予新的顺序,解决中间断层的问题
    """
    # 记录用到的标签
    old_labels_set = collections.defaultdict(list)
    for i, label in enumerate(pseudo_label):
        old_labels_set[label].append(i)
    new_labels_set = collections.defaultdict(list)
    # 赋予新的伪标签
    for new_label, old_label in enumerate(old_labels_set.keys()):
        new_labels_set[old_label] = new_label   # 将旧标签映射到新的标签,保证中间没有间断

    new_pseudo_label = []
    for label in pseudo_label:
        new_pseudo_label.append(new_labels_set[label])
    
    return new_pseudo_label

@torch.no_grad()
def generate_cluster_features(labels, features):
    centers = collections.defaultdict(list)
    for i, label in enumerate(labels):
        if label == -1:
            continue
        centers[labels[i]].append(features[i])
    centers = [
        torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
    ]
    centers = torch.stack(centers, dim=0)
    return centers

@torch.no_grad()
def generate_cluster_features_without_outliers(labels, features):
    centers = collections.defaultdict(list)
    for i, label in enumerate(labels):
        if label == -1:
            continue
        centers[labels[i]].append(features[i])
    centers = [
        torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys()) if len(centers[idx]) > 1
    ]
    centers = torch.stack(centers, dim=0)
    return centers

@torch.no_grad()
def re_ranking_for_instance(labels, memory_features, k1):
    N = memory_features[0].shape[0]
    similarity = memory_features[0].mm(memory_features[0].t())
    initial_rank = torch.argsort(similarity, dim=-1, descending=True)
    rank_k_matrix = torch.zeros_like(similarity)
    for i in range(N):
        forward_k_neigh_index = initial_rank[i,:k1+1] 
        backward_k_neigh_index = initial_rank[forward_k_neigh_index,:k1+1]
        fi = np.where(backward_k_neigh_index==i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate,:int(np.around(k1/2))+1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,:int(np.around(k1/2))+1]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index,k_reciprocal_index))> 2/3*len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index,candidate_k_reciprocal_index)
        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        rank_k_matrix[i, k_reciprocal_expansion_index] = 1
    

    label2set = collections.defaultdict(list)
    for i, label in enumerate(labels):
        if label == -1:
            continue
        label2set[label].append(i)

    label2vector = torch.zeros(len(label2set), N)
    for key, value in label2set.items():
        label2vector[key][torch.tensor(value)] = 1
    cluster_matrix = torch.zeros_like(rank_k_matrix)
    
    for i, label in enumerate(labels):
        cluster_matrix[i] = label2vector[label]

    m = rank_k_matrix.bool() & cluster_matrix.bool()
    uncertainty = m.sum(dim=-1) / torch.maximum(torch.tensor(1), torch.minimum(rank_k_matrix.sum(-1), cluster_matrix.sum(-1)))
    return uncertainty

@torch.no_grad()
def compute_label_transform_matrix(labels_t1, labels_t2):
    assert labels_t1.size(1) == labels_t2.size(1) # make sure sample num are equal
    sample_num = labels_t1.size(1)
    class_num_t1 = labels_t1.unique().size(0)
    class_num_t2 = labels_t2.unique().size(0)
    dual_labels = torch.cat((labels_t1, labels_t2),0).t()
    label_tran_mat = torch.zeros(class_num_t1, class_num_t2)
    for x in dual_labels:
        label_tran_mat[x[0].item(), x[1].item()] += 1
    return label_tran_mat

@torch.no_grad()
def compute_label_iou_matrix(labels_t1, labels_t2):
    class_num_t1 = labels_t1.unique().size(0)
    class_num_t2 = labels_t2.unique().size(0)
    dual_labels = torch.cat((labels_t1, labels_t2),0).t()
    label_union_mat_1 = torch.zeros(class_num_t1, class_num_t2)
    label_union_mat_2 = torch.zeros(class_num_t1, class_num_t2).t()
    for x in dual_labels:
        label_union_mat_1[x[0].item()] += 1
        label_union_mat_2[x[1].item()] += 1
    label_inter_mat = compute_label_transform_matrix(labels_t1, labels_t2)
    label_union_mat = label_union_mat_1 + label_union_mat_2.t() - label_inter_mat
    return label_inter_mat / label_union_mat

@torch.no_grad()
def extract_probabilities(features, centers, temp):
    features = F.normalize(features, p=2, dim=1)
    centers = F.normalize(centers, p=2, dim=1)
    logits = temp * features.mm(centers.t())
    prob = F.softmax(logits, 1)
    return prob

@torch.no_grad()
def compute_class_softlabels(labels_t1, labels_t2, matr_type="trans", distr_type="original"):
    assert labels_t1.size(1) == labels_t2.size(1) # make sure sample num are equal
    if matr_type == "trans":
        matr = compute_label_transform_matrix(labels_t1, labels_t2)
    else:
        matr = compute_label_iou_matrix(labels_t1, labels_t2)
    if distr_type=="original":
        return (matr.t() / matr.t().sum(0)).t()
    else:
        return torch.nn.functional.softmax(matr, 1)

@torch.no_grad()
def compute_sample_softlabels(labels_t1, labels_t2, matr_type="trans", distr_type="original"):
    class_softlabels = compute_class_softlabels(labels_t1, labels_t2, matr_type, distr_type)
    return torch.index_select(class_softlabels, 0, labels_t1[0])
