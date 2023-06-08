# --------------------------------------------------------
# Ke Yan,
# Imaging Biomarkers and Computer-Aided Diagnosis Laboratory (CADLab)
# National Institutes of Health Clinical Center,
# Apr 2019.
# This file contains codes to train and validate LesaNet.
# --------------------------------------------------------

import time
import torch
import numpy as np

from utils import AverageMeter, logger, debug, print_accs, clip_gradient
from load_save_utils import save_test_scores_to_file, save_ft_to_file, save_acc_to_file
from evaluate import score2label, compute_all_acc_wt, compute_all_acc_wt_th
from config import config, default
from my_algorithm import select_triplets_multilabel


def train(train_loader, model, criterions, optimizer, epoch):       #This function performs a training loop for one epoch 
    batch_time = AverageMeter()                                     #tracks batch processing
    data_time = AverageMeter()                                      #tracks data loading time
    losses = AverageMeter()                                         #tracks losses
    accs = AverageMeter()                                           #tracks accuracies

    # switch to train mode
    model.train()

    end = time.time()                                               #records the starting time of each epoch
    for i, (inputs, targets, infos) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # construct targets
        target_clsf, target_unc, target_ex = targets
        target_conf = (target_clsf + target_ex) > 0
        rhem_wt = torch.zeros_like(target_clsf).cuda()
        rhem_wt[target_conf] = 1.
        target_clsf = target_clsf.cuda()
        target_clsf_wt = 1-target_unc.cuda()

        # run model
        inputs = [input.cuda() for input in inputs]
        out = model(inputs)

        # compute losses
        emb = out['emb']
        A, P, N = select_triplets_multilabel(emb, target_clsf)                              #select triplets for the metric loss calculation
        loss_metric = criterions['metric'](A, P, N)                                         #calculates the metric loss

        prob1 = out['class_prob1']                                                          #gets the class_prob1 tensor from the out dictionary
        loss_ce1 = criterions['wce'](prob1, target_clsf, infos, wt=target_clsf_wt)          #calculated the weighted CE loss for prob1
        loss_rhem = criterions['rhem'](prob1, target_clsf, infos, wt=rhem_wt)               #calculates the RHEM for prob1
        if config.SCORE_PROPAGATION:
            prob2 = out['class_prob2']
            loss_ce2 = criterions['wce'](prob2, target_clsf, infos, wt=target_clsf_wt)      #calculates the weighted CE for prob2

            sub_losses = [loss_ce1, loss_rhem, loss_metric, loss_ce2]                       #corresponds the sub losses and weight names
            wts_names = ['CE_LOSS_WT_1', 'RHEM_LOSS_WT', 'TRIPLET_LOSS_WT', 'CE_LOSS_WT_2']
        else:
            sub_losses = [loss_ce1, loss_rhem, loss_metric]                                 #correspons the sub losses and weight names
            wts_names = ['CE_LOSS_WT_1', 'RHEM_LOSS_WT', 'TRIPLET_LOSS_WT']

        loss = 0
        wts = [eval('config.TRAIN.' + name1) for name1 in wts_names]                        #retrieves the weights based on the weight names
        for wt1, loss1 in zip(wts, sub_losses):
            loss += wt1 * loss1                                                             #calculates the weighted sum of the sub losses

        losses.update(loss.item())                                                          #updates the losses with the current loss

        # compute gradient and do SGD step
        optimizer.zero_grad()                                                               #clears the gradients
        loss.backward()                                                                     #uses back propogation
        clip_gradient(model, default.clip_gradient)                                         #clips the gradients to prevent for exploding
        optimizer.step()                                                                    #updates the parameters

        # measure accuracy
        if config.SCORE_PROPAGATION:
            prob_np = prob2.detach().cpu().numpy()
        else:
            prob_np = prob1.detach().cpu().numpy()

        pred_labels = score2label(prob_np, config.TEST.SCORE_PARAM)
        targets_np = target_clsf.detach().cpu().numpy()
        target_unc = target_unc.numpy()
        acc = compute_all_acc_wt(targets_np > 0, pred_labels, prob_np, target_unc == 0)[config.TEST.CRITERION]   

        accs.update(acc)                                                                   #updates the accuraces with the current accuracy

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % default.frequent == 0:
            crit = 'mean_pcF1' if config.TEST.CRITERION == 'mean_perclass_f1' else config.TEST.CRITERION
            msg = 'Epoch: [{0}][{1}/{2}] Time {batch_time.val:.1f} ' \
                  '({batch_time.avg:.1f}, {data_time.val:.1f})\t' \
                  .format(epoch, i, len(train_loader), batch_time=batch_time, data_time=data_time)
            msg += 'Loss {loss.val:.3f} ({loss.avg:.3f}){{'.format(loss=losses)
            for wt1, loss1 in zip(wts, sub_losses):
                msg += '%.3f*%.1f, ' % (loss1, wt1)
            msg += '}}\t{crit} {accs.val:.3f} ({accs.avg:.3f})'.format(
                crit=crit, accs=accs, ms=prob_np.max())
            logger.info(msg)


def validate(val_loader, model, use_val_th=False):
    batch_time = AverageMeter()                                                             #measures batch processing time
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():                   #disables gradient descent
        end = time.time()                   #record the start time of validation
        for i, (inputs, targets, infos) in enumerate(val_loader):
            if default.generate_features_all:
                logger.info('generating features, batch %d', i)
            filenames = [info[0] for info in infos]                         #extracts filenames from info
            lesion_idxs = [info[1] for info in infos]                       #extracts lesion indices from info
            inputs = [input.cuda() for input in inputs]                     #moves inputs to the GPU
            unc_targets = targets[1]                                        #extracts uncertain targets and actual targets
            targets = targets[0]

            # compute output
            out = model(inputs)                                             #does a forward pass through the model to get the output
            if config.SCORE_PROPAGATION:
                prob_np = out['class_prob2'].detach().cpu().numpy()         
                scores_np = out['class_score2'].detach().cpu().numpy()
            else:
                prob_np = out['class_prob1'].detach().cpu().numpy()
                scores_np = out['class_score1'].detach().cpu().numpy()

            target1 = targets.numpy() > 0                                       #converts targets to a numpy array, and checks if its >0
            pred_wt = unc_targets.numpy() == 0                                  #converts uncertain targets to 0
            if i == 0:
                target_all = target1                                            #initialization for target, prob, score, lesion_idx, pred
                prob_all = prob_np
                score_all = scores_np       
                lesion_idx_all = lesion_idxs
                pred_wt_all = pred_wt
                if default.generate_features_all:
                    ft_all = out['emb']
            else:
                target_all = np.vstack((target_all, target1))                   #vertically stacks all of the following
                prob_all = np.vstack((prob_all, prob_np))
                score_all = np.vstack((score_all, scores_np))
                pred_wt_all = np.vstack((pred_wt_all, pred_wt))
                lesion_idx_all.extend(lesion_idxs)
                if default.generate_features_all:
                    ft_all = np.vstack((ft_all, out['emb']))

        if default.generate_features_all:
            save_ft_to_file(ft_all)
            assert 0, 'all features have been generated and saved.'

        if config.TEST.USE_CALIBRATED_TH:
            accs, pred_label_all = compute_all_acc_wt_th(target_all, prob_all, pred_wt_all, use_val_th)
        else:
            pred_label_all = score2label(prob_all, config.TEST.SCORE_PARAM)
            accs = compute_all_acc_wt(target_all, pred_label_all, prob_all, pred_wt_all)

        # measure elapsed time
        batch_time.update(time.time() - end)                    #updates the batch time
        end = time.time()

        if i % default.frequent == 0:
            logger.info('Test: [{0}/{1}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        '{crit} {accs:.3f}'
                        .format(
                   i, len(val_loader), batch_time=batch_time, crit=config.TEST.CRITERION,
                   accs=accs[config.TEST.CRITERION]
            ))

        print_accs(accs)                                        #prints accuracy metrics
        accs['ex_neg'] = np.sum((target_all == 0) & pred_wt_all, axis=0)           
        if use_val_th:  # only save for test set not val set
            save_acc_to_file(accs, val_loader, 'all_terms')                                         #saves accuracy and test scores to file
        if default.mode == 'infer' and use_val_th:
            save_test_scores_to_file(score_all, pred_label_all, target_all, accs, lesion_idx_all)

    return accs


def adjust_learning_rate(optimizer, epoch):
    idx = np.where(epoch >= np.array([0]+default.lr_epoch))[0][-1]              #gets the index of learning rate epoch
    lr_factor = default.lr_factor ** idx                                        #computes the learning rate factor
    for param_group in optimizer.param_groups:                                  #iterates over the parameter groups in the optimizer
        if 'ori_lr' not in param_group.keys():  # first iteration               #saves the original learning rate
            param_group['ori_lr'] = param_group['lr']
        param_group['lr'] = param_group['ori_lr'] * lr_factor                   #adjusts the learning rate
    logger.info('learning rate factor %g' % lr_factor)                          #log the learning rate factor


    #summary: 
    #The train function performs a training loop for one epoch. It tracks batch processing time, data loading time, losses and accurarcies
    #It then sets the model to train mode, constructs targets based on the input data, and runs the model forward to get output
    #Then it computes different losses (metric loss, weighted CE, and RHEM)
    #Then it calculates the overall loss and updates the average loss, it also computes the gradients to update the parameters
    #Then it measures the accuracy by comparing the predicted labels to the actual output, and updates the accuracy

    #The validate function performs validation on data given. It first processes the batch of inputs to make the output probabilities
    #It calculates the predicted labels from the probabilities, and updates the overall targets, probabilities, scores, lesion indicces and weights
    #Then it calculates the accuract, and saves the metrics and test scores

    #The adjust learning rate function adjusts the learning rate of the optimizer based on the current epoch
    #it calculates the learning rate factor based on the epoch number, and adjusts the learning rate for every parameter in the optimizer
