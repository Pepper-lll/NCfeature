import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import random

eps = 1e-7

# CE and LDAM are supported

# If you would like to add other losses, please have a look at:
# Focal Loss: https://github.com/kaidic/LDAM-DRW
# CRD, PKT, and SP Related Part: https://github.com/HobbitLong/RepDistiller

def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, cls_num_list=None, weight=None, gamma=0.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight
    
    def _hook_before_epoch(self, epoch):
        pass

    def forward(self, output_logits, target):
        return focal_loss(F.cross_entropy(output_logits, target, reduction='none', weight=self.weight), self.gamma)

class CrossEntropyLoss(nn.Module):
    def __init__(self, cls_num_list=None, reweight_CE=False):
        super().__init__()
        if reweight_CE:
            idx = 1 # condition could be put in order to set idx
            betas = [0, 0.9999]
            effective_num = 1.0 - np.power(betas[idx], cls_num_list)
            per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            self.per_cls_weights = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False)
        else:
            self.per_cls_weights = None

    def to(self, device):
        super().to(device)
        if self.per_cls_weights is not None:
            self.per_cls_weights = self.per_cls_weights.to(device)
        
        return self

    def forward(self, output_logits, target): # output is logits
        return F.cross_entropy(output_logits, target, weight=self.per_cls_weights)

class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list=None, max_m=0.5, s=30, reweight_epoch=-1):
        super().__init__()
        if cls_num_list is None:
            # No cls_num_list is provided, then we cannot adjust cross entropy with LDAM.
            self.m_list = None
        else:
            self.reweight_epoch = reweight_epoch
            m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
            m_list = m_list * (max_m / np.max(m_list))
            m_list = torch.tensor(m_list, dtype=torch.float, requires_grad=False)
            self.m_list = m_list
            assert s > 0
            self.s = s
            if reweight_epoch != -1:
                idx = 1 # condition could be put in order to set idx
                betas = [0, 0.9999]
                effective_num = 1.0 - np.power(betas[idx], cls_num_list)
                per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
                per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
                self.per_cls_weights_enabled = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False)
            else:
                self.per_cls_weights_enabled = None
                self.per_cls_weights = None

    def to(self, device):
        super().to(device)
        if self.m_list is not None:
            self.m_list = self.m_list.to(device)

        if self.per_cls_weights_enabled is not None:
            self.per_cls_weights_enabled = self.per_cls_weights_enabled.to(device)

        return self

    def _hook_before_epoch(self, epoch):
        if self.reweight_epoch != -1:
            self.epoch = epoch

            if epoch > self.reweight_epoch:
                self.per_cls_weights = self.per_cls_weights_enabled
            else:
                self.per_cls_weights = None

    def get_final_output(self, output_logits, target):
        x = output_logits

        index = torch.zeros_like(x, dtype=torch.uint8, device=x.device)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.float()
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m * self.s

        final_output = torch.where(index, x_m, x)
        return final_output

    def forward(self, output_logits, target):
        if self.m_list is None:
            return F.cross_entropy(output_logits, target)
        
        final_output = self.get_final_output(output_logits, target)
        return F.cross_entropy(final_output, target, weight=self.per_cls_weights)

class RIDELoss(nn.Module):
    def __init__(self, cls_num_list=None, base_diversity_temperature=1.0, max_m=0.5, s=30, reweight=True, reweight_epoch=-1, 
        base_loss_factor=1.0, additional_diversity_factor=-0.2, reweight_factor=0.05):
        super().__init__()
        self.base_loss = F.cross_entropy
        self.base_loss_factor = base_loss_factor
        if not reweight:
            self.reweight_epoch = -1
        else:
            self.reweight_epoch = reweight_epoch

        # LDAM is a variant of cross entropy and we handle it with self.m_list.
        if cls_num_list is None:
            # No cls_num_list is provided, then we cannot adjust cross entropy with LDAM.

            self.m_list = None
            self.per_cls_weights_enabled = None
            self.per_cls_weights_enabled_diversity = None
        else:
            # We will use LDAM loss if we provide cls_num_list.

            m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
            m_list = m_list * (max_m / np.max(m_list))
            m_list = torch.tensor(m_list, dtype=torch.float, requires_grad=False)
            self.m_list = m_list
            self.s = s
            assert s > 0
            
            if reweight_epoch != -1:
                idx = 1 # condition could be put in order to set idx
                betas = [0, 0.9999]
                effective_num = 1.0 - np.power(betas[idx], cls_num_list)
                per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
                per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
                self.per_cls_weights_enabled = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False)
            else:
                self.per_cls_weights_enabled = None

            cls_num_list = np.array(cls_num_list) / np.sum(cls_num_list)
            C = len(cls_num_list)
            per_cls_weights = C * cls_num_list * reweight_factor + 1 - reweight_factor

            # Experimental normalization: This is for easier hyperparam tuning, the effect can be described in the learning rate so the math formulation keeps the same.
            # At the same time, the 1 - max trick that was previously used is not required since weights are already adjusted.
            per_cls_weights = per_cls_weights / np.max(per_cls_weights)

            assert np.all(per_cls_weights > 0), "reweight factor is too large: out of bounds"
            # save diversity per_cls_weights
            self.per_cls_weights_enabled_diversity = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False).cuda()

        self.base_diversity_temperature = base_diversity_temperature
        self.additional_diversity_factor = additional_diversity_factor

    def to(self, device):
        super().to(device)
        if self.m_list is not None:
            self.m_list = self.m_list.to(device)
        
        if self.per_cls_weights_enabled is not None:
            self.per_cls_weights_enabled = self.per_cls_weights_enabled.to(device)

        if self.per_cls_weights_enabled_diversity is not None:
            self.per_cls_weights_enabled_diversity = self.per_cls_weights_enabled_diversity.to(device)

        return self

    def _hook_before_epoch(self, epoch):
        if self.reweight_epoch != -1:
            self.epoch = epoch

            if epoch > self.reweight_epoch:
                self.per_cls_weights_base = self.per_cls_weights_enabled
                self.per_cls_weights_diversity = self.per_cls_weights_enabled_diversity
            else:
                self.per_cls_weights_base = None
                self.per_cls_weights_diversity = None

    def get_final_output(self, output_logits, target):
        x = output_logits

        index = torch.zeros_like(x, dtype=torch.uint8, device=x.device)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.float()
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m * self.s

        final_output = torch.where(index.bool(), x_m, x)
        return final_output

    def forward(self, output_logits, target, extra_info=None):
        if extra_info is None:
            return self.base_loss(output_logits, target)

        loss = 0

        # Adding RIDE Individual Loss for each expert
        for logits_item in extra_info['logits']:
            ride_loss_logits = output_logits if self.additional_diversity_factor == 0 else logits_item
            if self.m_list is None:
                loss += self.base_loss_factor * self.base_loss(ride_loss_logits, target)
            else:
                final_output = self.get_final_output(ride_loss_logits, target)
                loss += self.base_loss_factor * self.base_loss(final_output, target, weight=self.per_cls_weights_base)
            
            base_diversity_temperature = self.base_diversity_temperature

            if self.per_cls_weights_diversity is not None:
                diversity_temperature = base_diversity_temperature * self.per_cls_weights_diversity.view((1, -1))
                temperature_mean = diversity_temperature.mean().item()
            else:
                diversity_temperature = base_diversity_temperature
                temperature_mean = base_diversity_temperature
            
            output_dist = F.log_softmax(logits_item / diversity_temperature, dim=1)
            with torch.no_grad():
                # Using the mean takes only linear instead of quadratic time in computing and has only a slight difference so using the mean is preferred here
                mean_output_dist = F.softmax(output_logits / diversity_temperature, dim=1)
            
            loss += self.additional_diversity_factor * temperature_mean * temperature_mean * F.kl_div(output_dist, mean_output_dist, reduction='batchmean')
        
        return loss

class RIDELossWithDistill(nn.Module):
    def __init__(self, cls_num_list=None, additional_distill_loss_factor=1.0, distill_temperature=1.0, ride_loss_factor=1.0, **kwargs):
        super().__init__()
        self.ride_loss = RIDELoss(cls_num_list=cls_num_list, **kwargs)
        self.distill_temperature = distill_temperature

        self.ride_loss_factor = ride_loss_factor
        self.additional_distill_loss_factor = additional_distill_loss_factor

    def to(self, device):
        super().to(device)
        self.ride_loss = self.ride_loss.to(device)
        return self

    def _hook_before_epoch(self, epoch):
        self.ride_loss._hook_before_epoch(epoch)

    def forward(self, student, target=None, teacher=None, extra_info=None):
        output_logits = student
        if extra_info is None:
            return self.ride_loss(output_logits, target)

        loss = 0
        num_experts = len(extra_info['logits'])
        for logits_item in extra_info['logits']:
            loss += self.ride_loss_factor * self.ride_loss(output_logits, target, extra_info)
            distill_temperature = self.distill_temperature

            student_dist = F.log_softmax(student / distill_temperature, dim=1)
            with torch.no_grad():
                teacher_dist = F.softmax(teacher / distill_temperature, dim=1)
            
            distill_loss = F.kl_div(student_dist, teacher_dist, reduction='batchmean')
            distill_loss = distill_temperature * distill_temperature * distill_loss
            loss += self.additional_distill_loss_factor * distill_loss
        return loss

class RIDELossWithNC(nn.Module):
    def __init__(self, feat_dim=None, cls_num_list=None, ride_loss_factor=1.0, 
                 NC1_factor=0.05, NC2_factor=1., reg_factor=1e-4, **kwargs):
        super().__init__()
        self.feat_dim = feat_dim
        self.NC1_factor = NC1_factor
        self.NC2_factor = NC2_factor
        self.reg_factor = reg_factor
        self.NCLoss = NCLoss(num_classes=len(cls_num_list), feat_dim=feat_dim)
        
        self.ride_loss = RIDELoss(cls_num_list=cls_num_list, **kwargs)
        self.ride_loss_factor = ride_loss_factor

    def to(self, device):
        super().to(device)
        self.ride_loss = self.ride_loss.to(device)
        self.NCLoss = self.NCLoss.to(device)
        return self

    def _hook_before_epoch(self, epoch):
        self.ride_loss._hook_before_epoch(epoch)

    def forward(self, output_logits, target, extra_info=None, add_NCLoss=False):
        ride_loss = self.ride_loss(output_logits, target, extra_info)
        
        assert 'feat' in extra_info, "feat must be provided for NC loss"
        feats = extra_info['feat'] # (B, #experts, feat_dim)
        feats = feats.mean(dim=1) # (B, feat_dim)
        if add_NCLoss:
            NC_out = self.NCLoss(feats, target)
            NC1loss = NC_out['NC1loss'] * self.NC1_factor
            NC2loss = NC_out['NC1loss'] * self.NC2_factor
            CenterLoss = NC_out['CenterLoss'] * self.reg_factor
            return self.ride_loss_factor * ride_loss + NC1loss + NC2loss + CenterLoss
        else:
            return self.ride_loss_factor * ride_loss

class NCLoss(nn.Module):
    '''
    Modified Center loss, 1 / n_k ||h-miu||
    '''
    def __init__(self, num_classes, feat_dim):
        super(NCLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.means = nn.Parameter(torch.randn(num_classes, feat_dim))
        
    def to(self, device):
        super().to(device)
        self.means = self.means.to(device)
        return self
            
    def NC2Loss(self):
        '''
        NC2 loss: maximize the average minimum angle of each centered class mean
        Returns:
            loss: NC2 loss
            max_cosine: maximum cosine similarity of each centered class mean
        '''
        means = self.means
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

        return loss, max_cosine
    
    def CenterLoss(self):
        means = self.means
        reg = means.mean(dim=0).norm() ** 2
        return reg

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        device = x.device
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.means, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(x, self.means.t(), beta=1, alpha=-2)

        classes = torch.arange(self.num_classes).long().to(device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        D = torch.sum(dist, dim=0)
        N = mask.float().sum(dim=0) + 1e-10
        NC1loss = (D / N).clamp(min=1e-12, max=1e+12).sum() / self.num_classes
        
        NC2Loss, max_cosine = self.NC2Loss()
        CenterLoss = self.CenterLoss()

        out = {
            'NC1loss': NC1loss,
            'NC2Loss': NC2Loss,
            'CenterLoss': CenterLoss,
            'max_cosine': max_cosine,
            'class_means': self.means
        }
        return out