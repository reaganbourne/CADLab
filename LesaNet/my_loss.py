# --------------------------------------------------------
# Ke Yan,
# Imaging Biomarkers and Computer-Aided Diagnosis Laboratory (CADLab)
# National Institutes of Health Clinical Center,
# Apr 2019.
# This file contains codes of the losses of LesaNet.
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from time import time

from config import config, default
from utils import unique


class WeightedCeLoss(nn.Module):
    def __init__(self, pos_weight, neg_weight):                                                     #initializes the pos and neg weight, and converts weights to tensors
        super(WeightedCeLoss, self).__init__()
        self.pos_wt = torch.from_numpy(pos_weight).cuda()
        self.neg_wt = torch.from_numpy(neg_weight).cuda()

    def forward(self, prob, targets, infos, wt=None):
        prob = prob.clamp(min=1e-7, max=1-1e-7)                                                     #clamps it to ensure that the probability is valid
        if wt is None:
            wt1 = torch.ones_like(prob)                                                             #if there is no weight, then ones is assigned
        if config.TRAIN.CE_LOSS_WEIGHTED and self.pos_wt is not None:
            wt1 = wt * (targets.detach() * self.pos_wt + (1-targets.detach()) * self.neg_wt)        #calculates weighted loss

        loss = -torch.mean(wt1 * (torch.log(prob) * targets + torch.log(1-prob) * (1-targets)))

        return loss


class CeLossRhem(nn.Module):
    def __init__(self):
        super(CeLossRhem, self).__init__()

    def forward(self, prob, targets, infos, wt=None):
        if wt is None:
            wt = torch.ones_like(prob)
        prob = prob.clamp(min=1e-7, max=1-1e-7)
        with torch.no_grad():                                                                       #calculates the difference between probability and target
            prob_diff_wt = torch.abs((prob - targets) * wt) ** config.TRAIN.RHEM_POWER
            idx = torch.multinomial(prob_diff_wt.view(-1), config.TRAIN.RHEM_BATCH_SIZE, replacement=True)      #samples indeces based on weighted difference
            # hist = np.histogram(idx.cpu().numpy(), np.arange(torch.numel(prob)+1))[0]
            # hist = np.reshape(hist, prob.shape)
            # pos = np.where(hist == np.max(hist))
            # row = pos[0][0]
            # col = pos[1][0]
            # print np.max(hist), prob[row, col].item(), targets[row, col].item(), \
            #     default.term_list[col], int(self.pos_wt[col].item()), infos[row][0]#, prob_diff_wt.mean(0)[col].item()

        targets = targets.view(-1)[idx]
        prob = prob.view(-1)[idx]
        loss_per_smp = - (torch.log(prob) * targets + torch.log(1-prob) * (1-targets))               #calculates loss per sample
        loss = loss_per_smp.mean()                                                                   #calculates mean loss

        return loss

#WeightedCeLoss is a loss function used for weighted binary cross entropy loss
#During forward pass it computes the weighted CE between probabilities and targets
#The loss is calculated by taking the negative mean of the weighted CE term

#CeLossRhem is used for a RHEM cross-entropy (ranking with high error magnitude)
#During the forward pass it computes the RHEM loss between predicted probabilities and targets
#First it makes sure that the probabilities are within a valid range
#Then it calculates a weighted difference between probabilities and targets using the weights
#The function samples indices based on the weighted difference and selects the corresponding targets and probabilities 
#The loss per sample is calculated as the negative log of the targets weighted by