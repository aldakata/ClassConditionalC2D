import os
import pickle

import numpy as np
from pandas.core import base
import torch
from sklearn.mixture import GaussianMixture

from train_uncertainty import warmup, train

from processing_utils import save_net_optimizer_to_ckpt

from uncertainty_utils import log_loss, gmm_pred
import sys

import tables
import datetime

def enable_dropout(net):
    """ Function to enable the dropout layers during test-time """
    for m in net.modules():
        if m.__class__.__name__.startswith('Drop'):
            m.train()

def save_losses(input_loss, exp):
    name = './stats/cifar100/losses{}.pcl'
    nm = name.format(exp)
    if os.path.exists(nm):
        loss_history = pickle.load(open(nm, "rb"))
    else:
        loss_history, clean_history = [], []
    loss_history.append(input_loss)
    pickle.dump(loss_history, open(nm, "wb"))

def eval_train(model, eval_loader, CE, all_loss, epoch, net, device, r, stats_log, loss_log, sample_entropy_log, sample_pred_log, gmm_log, subnet_loss_log, mcdo_passes = 3):
    model.eval()
    enable_dropout(model)
    epsilon = sys.float_info.min
    losses = torch.zeros(size=(50000, mcdo_passes))
    losses_clean = torch.zeros(size=(50000, mcdo_passes))
    softmaxs = torch.zeros(size=(50000, 10, mcdo_passes), device=device)
    targets_all = torch.zeros(50000, device=device)
    targets_all_clean = torch.zeros(50000, device=device)

    with torch.no_grad():
        for batch_idx, (inputs, _, targets, index, targets_clean) in enumerate(eval_loader):
            inputs, targets, targets_clean = inputs.to(device), targets.to(device), targets_clean.to(device)
            targets_all[index[0]:index[-1]+1] = targets
            targets_all_clean[index[0]:index[-1]+1] = targets_clean
            for i in range(mcdo_passes):
                outputs = model(inputs)
                loss = CE(outputs, targets)
                clean_loss = CE(outputs, targets_clean)
                softmax = torch.softmax(outputs, dim=1) # shape (n_samples, n_classes)
                for b in range(inputs.size(0)):
                    losses[index[b], i] = loss[b]
                    losses_clean[index[b], i] = clean_loss[b]
                    softmaxs[index[b], :, i] = softmax[b] # shape (n_samples, n_classes, n_mcdo_passes)

    # Per sample uncertainty.
    sample_mean_over_mcdo = torch.mean(softmaxs, dim=2) # shape (n_samples, n_classes)
    sample_variance_over_mcdo = torch.var(softmaxs, dim=2) # shape (n_samples, n_classes)
    sample_entropy = -torch.sum(sample_mean_over_mcdo*torch.log(sample_mean_over_mcdo + epsilon), axis=-1) # shape (n_samples,)

    # Per class uncertainty.
    sample_class_variance = torch.gather(sample_variance_over_mcdo, 1, targets_all.unsqueeze(-1).long()).squeeze() # TODO build targets.
    class_variance = torch.tensor([torch.mean(sample_class_variance[targets_all==c]).item() for c in range(10)]) # where 10 is num_classes

    # True Clean / Noisy
    clean_indices = targets_all_clean == targets_all

    # Vanilla Loss
    optimist_losses, indices = torch.min(losses, dim=1) # MOD: We stick with the worst loss for each sample
    optimist_losses_clean = torch.gather(losses_clean, 1, indices.unsqueeze(-1))
    pessimist_losses, indices = torch.max(losses, dim=1) # MOD: We stick with the worst loss for each sample
    pessimist_losses_clean = torch.gather(losses_clean, 1, indices.unsqueeze(-1))
    losses_clean = torch.cat((losses_clean, pessimist_losses_clean, optimist_losses_clean), dim=1)
    losses = torch.cat((losses, pessimist_losses.unsqueeze(-1), optimist_losses.unsqueeze(-1)), dim=1)
    losses_chosen = losses[:, 0] # Only choose the pass without dropout
    losses_chosen = (losses_chosen - losses_chosen.min()) / (losses_chosen.max() - losses_chosen.min())

    all_loss.append(losses_chosen)
    history = torch.stack(all_loss)

    if r >= 0.9:  # average loss over last 5 epochs to improve convergence stability
        input_loss = history[-5:].mean(0)
        input_loss = input_loss.reshape(-1, 1)
    else:
        input_loss = losses_chosen.reshape(-1, 1)

    # exp = '_std_tpc_oracle'
    # save_losses(input_loss, exp)

    gmm, clean_idx, noisy_idx, pred, prob = gmm_pred(input_loss) # uncertainty_utils

    # if not epoch%15:
    #     with tables.open_file(sample_entropy_log, mode='a') as f:
    #         f.root.data.append([list(sample_entropy.cpu().numpy())])

    #     with tables.open_file(sample_pred_log, mode='a') as f:
    #         f.root.data.append([list(pred)])

    #     for i in range(mcdo_passes+2):
    #         with tables.open_file(subnet_loss_log.format(i), mode='a') as f:
    #             f.root.data.append([list(losses[:,i].cpu().numpy())])
    clean_indices = clean_indices.cpu().numpy()
    tp_tn = clean_indices == pred
    fp_fn = clean_indices != pred
    tp = np.logical_and(tp_tn, clean_indices)
    tn = np.logical_and(tp_tn, ~clean_indices)
    fp = np.logical_and(fp_fn, ~clean_indices)
    fn = np.logical_and(fp_fn, clean_indices)

    stats_log.write('Epoch {}: (net {}): CLEAN GMM results: {} with weight: {} with covariance: {}\t NOISY GMM results: {} with weight: {} with covariance: {}\t'
                    'clean_entropy mean: {} | std: {} noisy_entropy mean: {} | std :{}\t labeled_entropy mean: {} | std: {} unlabeled_entropy mean : {} | std: {}\t'
                    'TP_TN  mean: {} std{}\t FP_FN mean: {} std: {}\tclass_variance: {}\n'
                    ''.format(epoch, net, gmm.means_[clean_idx], gmm.weights_[clean_idx], gmm.covariances_[clean_idx],
                                                 gmm.means_[noisy_idx], gmm.weights_[noisy_idx], gmm.covariances_[noisy_idx], 
                                                 sample_entropy[clean_indices].mean().item(), sample_entropy[clean_indices].std().item(),
                                                 sample_entropy[~clean_indices].mean().item(), sample_entropy[~clean_indices].std().item(), 
                                                 sample_entropy[pred].mean().item(), sample_entropy[pred].std().item(),
                                                 sample_entropy[~pred].mean().item(), sample_entropy[~pred].std().item(),
                                                 sample_entropy[tp_tn].mean().item(), sample_entropy[tp_tn].std().item(),
                                                 sample_entropy[fp_fn].mean().item(), sample_entropy[fp_fn].std().item(),
                                                 class_variance.tolist()))
    stats_log.flush()

    loss_log.write('{},{},{},{},{},{},{},{},{}\n'.format(epoch, losses_clean[pred].mean(dim=0).numpy(), losses_clean[pred].std(dim=0).numpy(),
                                                        losses_clean[~pred].mean(dim=0).numpy(), losses_clean[~pred].std(dim=0).numpy(),
                                                        losses[clean_indices].mean(dim=0).numpy(), losses[clean_indices].std(dim=0).numpy(),
                                                        losses[~clean_indices].mean(dim=0).numpy(), losses[~clean_indices].std(dim=0).numpy()))
    loss_log.flush()
    
    mcdo_gmm_accuracies = [(gmm_pred(loss, True)[-2] == clean_indices).sum()/50000 for loss in torch.transpose(losses, 0, 1)]
    gmm_log.write(f'{epoch}:{mcdo_gmm_accuracies}\n')
    gmm_log.flush()

    return prob, all_loss, losses_clean, class_variance

def run_test(epoch, net1, net2, test_loader, device, test_log):
    net1.eval()
    net2.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs1 = net1(inputs)
            outputs2 = net2(inputs)
            outputs = outputs1 + outputs2
            _, predicted = torch.max(outputs, 1)

            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()
    acc = 100. * correct / total
    print("\n| Test Epoch #%d\t Accuracy: %.2f%%\n" % (epoch, acc))
    test_log.write('Epoch:%d   Accuracy:%.2f\n' % (epoch, acc))
    test_log.flush()

def run_train_loop_mcdo(net1, optimizer1, sched1, net2, optimizer2, sched2, criterion, CEloss, CE, loader, p_threshold,
                   warm_up, num_epochs, all_loss, batch_size, num_class, device, lambda_u, lambda_c, T, alpha, noise_mode,
                   dataset, r, conf_penalty, stats_log, loss_log1, loss_log2, test_log, sample_entropy_log, sample_pred_log, gmm_log, subnet_loss_log, ckpt_path, resume_epoch):
    for epoch in range(resume_epoch, num_epochs + 1):
        test_loader = loader.run('test')
        eval_loader = loader.run('eval_train')

        if epoch < warm_up:
            warmup_trainloader = loader.run('warmup')
            print('Warmup Net1')
            warmup(epoch, net1, optimizer1, warmup_trainloader, CEloss, conf_penalty, device, dataset, r, num_epochs,
                   noise_mode)
            print('\nWarmup Net2')
            warmup(epoch, net2, optimizer2, warmup_trainloader, CEloss, conf_penalty, device, dataset, r, num_epochs,
                   noise_mode)

            # prob1, all_loss[0], losses_clean1, class_variance1 = eval_train(net1, eval_loader, CE, all_loss[0], epoch, 1, device, r,
            #                                                 stats_log, loss_log1)
            # prob2, all_loss[1], losses_clean2, class_variance2 = eval_train(net2, eval_loader, CE, all_loss[1], epoch, 2, device, r,
            #                                                stats_log)

            # p_thr2 = np.clip(p_threshold, prob2.min() + 1e-5, prob2.max() - 1e-5)
            # pred2 = prob2 > p_thr2

            # loader.run('train', pred2, prob2)  # count metrics
        else:
            print('Train Net1')
            begin_time = datetime.datetime.now()
            prob2, all_loss[1], losses_clean2, class_variance2 = eval_train(net2, eval_loader, CE, all_loss[1], epoch, 2, device, r, stats_log, loss_log2, sample_entropy_log, sample_pred_log, gmm_log, subnet_loss_log)
            end_time = datetime.datetime.now()
            print(f'CoDivide elapsed time: {end_time-begin_time}')
            p_thr2 = np.clip(p_threshold, prob2.min() + 1e-5, prob2.max() - 1e-5)
            pred2 = prob2 > p_thr2

            labeled_trainloader, unlabeled_trainloader = loader.run('train', pred2, prob2)  # co-divide
            train(epoch, net1, net2, criterion, optimizer1, labeled_trainloader, unlabeled_trainloader, lambda_u, lambda_c,
                  batch_size, num_class, device, T, alpha, warm_up, dataset, r, noise_mode, num_epochs, class_variance2)  # train net1

            print('\nTrain Net2')
            begin_time = datetime.datetime.now()
            prob1, all_loss[0], losses_clean1, class_variance1 = eval_train(net1, eval_loader, CE, all_loss[0], epoch, 1, device, r, stats_log, loss_log1, sample_entropy_log, sample_pred_log, gmm_log, subnet_loss_log)
            end_time = datetime.datetime.now()
            print(f'CoDivide elapsed time: {end_time-begin_time}')

            p_thr1 = np.clip(p_threshold, prob1.min() + 1e-5, prob1.max() - 1e-5)
            pred1 = prob1 > p_thr1

            labeled_trainloader, unlabeled_trainloader = loader.run('train', pred1, prob1)  # co-divide
            train(epoch, net2, net1, criterion, optimizer2, labeled_trainloader, unlabeled_trainloader, lambda_u, lambda_c,
                  batch_size, num_class, device, T, alpha, warm_up, dataset, r, noise_mode, num_epochs, class_variance1)  # train net2

        if not epoch%2 or True:
            print(f'[ SAVING MODELS] EPOCH: {epoch} PATH: {ckpt_path}')
            save_net_optimizer_to_ckpt(net1, optimizer1, f'{ckpt_path}/{epoch}_1.pt')
            save_net_optimizer_to_ckpt(net2, optimizer2, f'{ckpt_path}/{epoch}_2.pt')
        
        run_test(epoch, net1, net2, test_loader, device, test_log)

        sched1.step()
        sched2.step()
    torch.save(net1.state_dict(), './final_checkpoints/final_checkpoint.pth.tar')
