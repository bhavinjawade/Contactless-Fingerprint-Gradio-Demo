from pytorch_metric_learning import losses 
import torch
import torch.nn as nn
import torch.nn.init
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.weight_norm import weight_norm
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm
import numpy as np
import os
import torch.nn.functional as F

torch.autograd.set_detect_anomaly(True)

def get_MSloss():
    msloss = losses.MultiSimilarityLoss(alpha=2, beta=50, base=0.5)
    return msloss

def get_Arcface(num_classes, embedding_size):
    msloss = losses.ArcFaceLoss(num_classes, embedding_size, margin=28.6, scale=64)
    return msloss

def get_ProxyAnchor(num_classes, embedding_size):
    proxyanchor = losses.ProxyAnchorLoss(num_classes, embedding_size, margin = 0.1, alpha = 32)
    return proxyanchor


class DualMSLoss(nn.Module):
    """
    Compute contrastive loss
    """
    def __init__(self, margin=0, max_violation=False):
        super(DualMSLoss, self).__init__()
        self.margin = margin
        self.max_violation = max_violation
        self.thresh = 0.5
        self.margin = 0.1
        self.regions = 36
        self.scale_pos = 2.0
        self.scale_neg = 40.0
        self.scale_pos_p = 2.0
        self.scale_neg_p = 40.0

    def ms_sample(self,sim_mat,label):
        pos_exp = torch.exp(-self.scale_pos*(sim_mat-self.thresh))
        neg_exp = torch.exp( self.scale_neg*(sim_mat-self.thresh))
        pos_mask = torch.eq(label.view(-1,1)-label.view(1,-1),0.0).float().cuda()
        neg_mask = 1-pos_mask
        P_sim = torch.where(pos_mask == 1,sim_mat,torch.ones_like(pos_exp)*1e16)
        N_sim = torch.where(neg_mask == 1,sim_mat,torch.ones_like(neg_exp)*-1e16)
        min_P_sim,_ = torch.min(P_sim,dim=1,keepdim=True)
        max_N_sim,_ = torch.max(N_sim,dim=1,keepdim=True)
        hard_P_sim = torch.where(P_sim - self.margin < max_N_sim,pos_exp,torch.zeros_like(pos_exp)).sum(dim=-1)
        hard_N_sim = torch.where(N_sim + self.margin > min_P_sim,neg_exp,torch.zeros_like(neg_exp)).sum(dim=-1)
        pos_loss = torch.log(1+hard_P_sim).sum()/self.scale_pos
        neg_loss = torch.log(1+hard_N_sim).sum()/self.scale_neg
        
        return pos_loss + neg_loss

    def forward(self, x_contactless, x_contactbased, labels):
        sim_mat = F.linear(self.l2_norm(x_contactless), self.l2_norm(x_contactbased))
        loss1 = self.ms_sample(sim_mat, labels).cuda() + self.ms_sample(sim_mat.t(), labels).cuda()

        sim_mat = F.linear(self.l2_norm(x_contactless), self.l2_norm(x_contactless))
        loss2 = self.ms_sample(sim_mat, labels).cuda() + self.ms_sample(sim_mat.t(), labels).cuda()

        sim_mat = F.linear(self.l2_norm(x_contactbased), self.l2_norm(x_contactbased))
        loss3 = self.ms_sample(sim_mat, labels).cuda() + self.ms_sample(sim_mat.t(), labels).cuda()

        return loss1 + loss2 + loss3

    def l2_norm(self, input):
        input_size = input.size()
        buffer = torch.pow(input, 2)
        normp = torch.sum(buffer, 1).add_(1e-12)
        norm = torch.sqrt(normp)
        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)
        
        return output