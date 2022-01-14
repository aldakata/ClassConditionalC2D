import os
import pickle

import numpy as np
import torch
from sklearn.mixture import GaussianMixture

from train import warmup, train

from processing_utils import save_net_optimizer_to_ckpt
from uncertainty_utils import log_loss, gmm_pred, ccgmm_pred, or_ccgmm, and_ccgmm, mean_ccgmm
from constants import OR_CCGMM, AND_CCGMM, CCGMM, GMM, MEAN_CCGMM

def save_losses(input_loss, exp):
    name = './stats/cifar100/losses{}.pcl'
    nm = name.format(exp)
    if os.path.exists(nm):
        loss_history = pickle.load(open(nm, "rb"))
    else:
        loss_history, clean_history = [], []
    loss_history.append(input_loss)
    pickle.dump(loss_history, open(nm, "wb"))


def eval_train(model, eval_loader, CE, all_loss, epoch, net, device, r, stats_log, p_threshold, division):
    model.eval()
    losses = torch.zeros(50000)
    losses_clean = torch.zeros(50000)
    targets_all = torch.zeros(50000, device=device)

    with torch.no_grad():
        for batch_idx, (inputs, _, targets, index, targets_clean) in enumerate(eval_loader):
            inputs, targets, targets_clean = inputs.to(device), targets.to(device), targets_clean.to(device)
            outputs = model(inputs)
            loss = CE(outputs, targets)
            clean_loss = CE(outputs, targets_clean)
            for b in range(inputs.size(0)):
                losses[index[b]] = loss[b]
                losses_clean[index[b]] = clean_loss[b]
                targets_all[index[b]] = targets[b]
                
    losses = (losses - losses.min()) / (losses.max() - losses.min())
    all_loss.append(losses)

    history = torch.stack(all_loss)

    if r >= 0.9:  # average loss over last 5 epochs to improve convergence stability
        input_loss = history[-5:].mean(0)
        input_loss = input_loss.reshape(-1, 1)
    else:
        input_loss = losses.reshape(-1, 1)


    if division == GMM:
        l = input_loss
        gaussian_mixture = gmm_pred
    elif division == CCGMM:
        l = input_loss
        gaussian_mixture = ccgmm_pred
    elif division == OR_CCGMM:
        l = losses
        gaussian_mixture = or_ccgmm
    elif division == AND_CCGMM:
        l = input_loss
        gaussian_mixture = and_ccgmm
    elif division == MEAN_CCGMM:
        l = input_loss
        gaussian_mixture = mean_ccgmm
    

    print(f'DIVISION: {division}')
    prob, pred = gaussian_mixture(l, targets_all, p_threshold) # uncertainty_utils


    return prob, all_loss, losses_clean, pred


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


def run_train_loop(net1, optimizer1, sched1, net2, optimizer2, sched2, criterion, CEloss, CE, loader, p_threshold,
                   warm_up, num_epochs, all_loss, batch_size, num_class, device, lambda_u, T, alpha, noise_mode,
                   dataset, r, conf_penalty, stats_log, loss_log, test_log, ckpt_path, resume_epoch, division):
    for epoch in range(resume_epoch, num_epochs + 1):
        test_loader = loader.run('test')
        eval_loader = loader.run('eval_train')

        if epoch < warm_up:
            warmup_trainloader = loader.run('warmup')
            print('\nWarmup Net1')
            warmup(epoch, net1, optimizer1, warmup_trainloader, CEloss, conf_penalty, device, dataset, r, num_epochs,
                   noise_mode)

            print('\nWarmup Net2')
            warmup(epoch, net2, optimizer2, warmup_trainloader, CEloss, conf_penalty, device, dataset, r, num_epochs,
                   noise_mode)

            # prob1, all_loss[0], losses_clean1 = eval_train(net1, eval_loader, CE, all_loss[0], epoch, 1, device, r,
            #                                                stats_log)
            # prob2, all_loss[1], losses_clean2 = eval_train(net2, eval_loader, CE, all_loss[1], epoch, 2, device, r,
            #                                                stats_log)

            # p_thr2 = np.clip(p_threshold, prob2.min() + 1e-5, prob2.max() - 1e-5)
            # pred2 = prob2 > p_thr2

            # loss_log.write('{},{},{},{},{}\n'.format(epoch, losses_clean2[pred2].mean(), losses_clean2[pred2].std(),
            #                                          losses_clean2[~pred2].mean(), losses_clean2[~pred2].std()))
            # loss_log.flush()
            # loader.run('train', pred2, prob2)  # count metrics
        else:
            print('Train Net1')
            prob2, all_loss[1], losses_clean2, pred2 = eval_train(net2, eval_loader, CE, all_loss[1], epoch, 2, device, r,
                                                           stats_log, p_threshold, division)

            loss_log.write('{},{},{},{},{}\n'.format(epoch, losses_clean2[pred2].mean(), losses_clean2[pred2].std(),
                                                     losses_clean2[~pred2].mean(), losses_clean2[~pred2].std()))
            loss_log.flush()

            labeled_trainloader, unlabeled_trainloader = loader.run('train', pred2, prob2)  # co-divide
            train(epoch, net1, net2, criterion, optimizer1, labeled_trainloader, unlabeled_trainloader, lambda_u,
                  batch_size, num_class, device, T, alpha, warm_up, dataset, r, noise_mode, num_epochs)  # train net1

            print('\nTrain Net2')
            prob1, all_loss[0], losses_clean1, pred1 = eval_train(net1, eval_loader, CE, all_loss[0], epoch, 1, device, r,
                                                           stats_log, p_threshold, division)

            labeled_trainloader, unlabeled_trainloader = loader.run('train', pred1, prob1)  # co-divide
            train(epoch, net2, net1, criterion, optimizer2, labeled_trainloader, unlabeled_trainloader, lambda_u,
                  batch_size, num_class, device, T, alpha, warm_up, dataset, r, noise_mode, num_epochs)  # train net2

        if not epoch%5:
            save_net_optimizer_to_ckpt(net1, optimizer1, f'{ckpt_path}/1.pt')
            save_net_optimizer_to_ckpt(net2, optimizer2, f'{ckpt_path}/2.pt')

        run_test(epoch, net1, net2, test_loader, device, test_log)

        sched1.step()
        sched2.step()
