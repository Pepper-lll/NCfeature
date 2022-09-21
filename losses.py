import math
from turtle import forward
from pyparsing import alphas
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

def compute_adjustment(train_loader, tro):
    """compute the base probabilities"""
    label_freq = {}
    for i, (inputs, target, index) in enumerate(train_loader):
        target = target.cuda()
        for j in target:
            key = int(j.item())
            label_freq[key] = label_freq.get(key, 0) + 1
    label_freq = dict(sorted(label_freq.items()))
    label_freq_array = np.array(list(label_freq.values()))
    label_freq_array = label_freq_array / label_freq_array.sum()
    adjustments = np.log(label_freq_array ** tro + 1e-12)
    adjustments = torch.from_numpy(adjustments)
    adjustments = adjustments.cuda()
    return adjustments

def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = (1 -  p) ** gamma * input_values
    return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        return focal_loss(F.cross_entropy(input, target, reduction='none', weight=self.weight), self.gamma)

class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s*output, target, weight=self.weight)

class NC1Loss(nn.Module):
    '''
    Modified Center loss, 1 / n_k ||h-miu||
    '''
    def __init__(self, num_classes=10, feat_dim=128, use_gpu=True):
        super(NC1Loss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.means = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.means = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.means, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(x, self.means.t(), beta=1, alpha=-2)

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        D = torch.sum(dist, dim=0)
        N = mask.float().sum(dim=0) + 1e-10
        # loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        loss = (D / N).clamp(min=1e-12, max=1e+12).sum() / self.num_classes

        return loss, self.means

    
class NC1Loss_1(nn.Module):
    '''
    Modified Center loss, 1 / n_k**2 ||h-miu||
    '''
    def __init__(self, num_classes=10, feat_dim=128, use_gpu=True):
        super(NC1Loss_1, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.means = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.means = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.means, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(x, self.means.t(), beta=1, alpha=-2)

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        D = torch.sum(dist, dim=0)
        N = mask.float().sum(dim=0) + 1e-5
        N = N ** 2
        # print()
        # loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        loss = (D/N).clamp(min=1e-12, max=1e+12).sum() / self.num_classes

        return loss, self.means


def NC2Loss(means):
    '''
    NC2 loss v0: maximize the average minimum angle of each centered class mean
    '''
    g_mean = means.mean(dim=0)
    centered_mean = means - g_mean
    means_ = F.normalize(centered_mean, p=2, dim=1)
    cosine = torch.matmul(means_, means_.t())
    # make sure that the diagnonal elements cannot be selected
    cosine = cosine - 2. * torch.diag(torch.diag(cosine))
    max_cosine = cosine.max().clamp(-0.99999, 0.99999)
    # print('min angle:', min_angle)
    # maxmize the minimum angle
    # dim=1 means the maximum angle of the other class to each class
    loss = -torch.acos(cosine.max(dim=1)[0].clamp(-0.99999, 0.99999)).mean()
    # loss = cosine.max(dim=1)[0].clamp(-0.99999, 0.99999).mean() + 1. / (means.size(0)-1)

    return loss, max_cosine

def NC2Loss_v1(means):
    '''
    NC2 loss v1: maximize the minimum angle of centered class means
    '''
    g_mean = means.mean(dim=0)
    centered_mean = means - g_mean
    means_ = F.normalize(centered_mean, p=2, dim=1)
    cosine = torch.matmul(means_, means_.t())
    # make sure that the diagnonal elements cannot be selected
    cosine = cosine - 2. * torch.diag(torch.diag(cosine))
    max_cosine = cosine.max().clamp(-0.99999, 0.99999)
    # print('min angle:', min_angle)
    # maxmize the minimum angle
    # dim=1 means the maximum angle of the other class to each class
    loss = -torch.acos(max_cosine)
    min_angle = math.degrees(torch.acos(max_cosine.detach()).item())
    # loss = cosine.max(dim=1)[0].clamp(-0.99999, 0.99999).mean() + 1. / (means.size(0)-1)

    return loss, max_cosine

def NC2Loss_v2(means):
    '''
    NC2 loss: make the cosine of any pair of class-means be close to -1/(C-1))
    '''
    C = means.size(0)
    g_mean = means.mean(dim=0)
    centered_mean = means - g_mean
    means_ = F.normalize(centered_mean, p=2, dim=1)
    cosine = torch.matmul(means_, means_.t())
    # make sure that the diagnonal elements cannot be selected
    cosine_ = cosine - 2. * torch.diag(torch.diag(cosine))
    max_cosine = cosine_.max().clamp(-0.99999, 0.99999)
    cosine = cosine_ + (1. - 1/(C-1)) * torch.diag(torch.diag(cosine))
    # print('min angle:', min_angle)
    # maxmize the minimum angle
    loss = cosine.norm()
    # loss = -torch.acos(cosine.max(dim=1)[0].clamp(-0.99999, 0.99999)).mean()
    # loss = cosine.max(dim=1)[0].clamp(-0.99999, 0.99999).mean() + 1. / (means.size(0)-1)

    return loss, max_cosine

class ARBLoss(nn.Module):
    def __init__(self, reduction: str = 'mean'):
        super(ARBLoss, self).__init__()
        self.reduction = reduction
        
    def forward(self, output, y):
        n, C = output.size()
        class_data_num = []
        for l in range(max(y)+1):
            class_data_num.append(len(y[y == l]))
        # print(class_data_num)
        w = torch.zeros_like(y)
        for i in range(y.size(0)):
            w[i] = float(class_data_num[y[i]])
        w = w.unsqueeze(1)
        w = w.expand(n, C)
        # print(w)
        tmp = n/w * output
        logit = output / tmp.sum(dim=1).unsqueeze(1)
        log_logit = torch.log(logit)
        cri = nn.NLLLoss(reduction=self.reduction)
        loss = cri(log_logit, y)
        return loss
    

def zero_center(means):
    g_mean = means.mean(dim=0)
    reg = g_mean.norm() ** 2
    return reg

class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss