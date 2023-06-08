# --------------------------------------------------------
# Ke Yan,
# Imaging Biomarkers and Computer-Aided Diagnosis Laboratory (CADLab)
# National Institutes of Health Clinical Center,
# Apr 2019.
# This file contains codes of certain algorithmic functions.
# --------------------------------------------------------

import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform

from config import config, default


def select_triplets_multilabel(emb, target):
    target = target.detach().cpu().numpy()                          #converts target tensor to numpy array and detach it from CPU
    num_smp = len(target)                                           #gets the number of samples
    inters = np.matmul(target, target.T)                            #calculates the intersection between target samples
    iou = 1 - squareform(pdist(target, 'jaccard'))                  #calculates the jaccard distance between target labels
    sim = inters * iou + np.diag(np.nan*np.ones((num_smp,)))        #calculates the similarity matrix by combining the intersection and jaccard distance
    # sim = inters + np.diag(np.nan*np.ones((num_smp,)))

    triplets = []                                                   #initializes a list to store the triplets
    for p in range(config.TRAIN.NUM_TRIPLET):                       #generates the triplets
        a = np.random.choice(num_smp)                               #sets the anchor randomly
        p_candidates = np.where(sim[a] >= config.TRAIN.SIMILAR_LABEL_THRESHOLD)[0]              #finds a positive candidate thats >=threshold
        # p = np.random.choice(np.setdiff1d(range(num_smp), [a]))
        if len(p_candidates) == 0:                                  #if there are no positive candidates, then find one with the highest similarity
            p = np.nanargmax(sim[a])
        else:
            p = np.random.choice(p_candidates)          
        if sim[a, p] - np.nanmin(sim[a]) <= config.TRAIN.DISSIMILAR_LABEL_THRESHOLD: #check if similarity between anchor and positive sample <= threshold
            n = np.nanargmin(sim[a])
            # if sim[a,p] == sim[a,n]:
            #     continue
        else:
            n_candidates = np.where(sim[a] < sim[a, p] - config.TRAIN.DISSIMILAR_LABEL_THRESHOLD)[0]        #find negative candidates based on dissimalarity threshold
            n = np.random.choice(n_candidates)
        triplets.append([a, p, n])                                  #append the triplet to the triplets list

    A = emb[[triplet[0] for triplet in triplets]]                   #selects the corresponding embeddings
    P = emb[[triplet[1] for triplet in triplets]]
    N = emb[[triplet[2] for triplet in triplets]]
    return A, P, N                                                  #returns the embeddings

#This code selects embeddings corresponding to the anchor, positive and negative samples, and returns them as triplets
