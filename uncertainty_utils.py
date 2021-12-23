import numpy as np
from sklearn.mixture import GaussianMixture
import torch

def log_loss(epoch, losses_clean, pred, loss_log):
    loss_log.write('{},{},{},{},{}\n'.format(epoch, losses_clean[pred].mean(dim=1), losses_clean[pred].std(dim=1),
                                                        losses_clean[~pred].mean(dim=1), losses_clean[~pred].std(dim=1)))
    loss_log.flush()

def gmm_pred(loss, targets, thr, reshape = False):
    if reshape:
        loss = loss.reshape(-1,1)
    gmm = GaussianMixture(n_components=2, max_iter=200, tol=1e-2, reg_covar=5e-4)
    gmm.fit(loss)

    clean_idx, noisy_idx = gmm.means_.argmin(), gmm.means_.argmax()

    prob = gmm.predict_proba(loss)
    prob = prob[:, clean_idx]

    p_thr = np.clip(thr, prob.min() + 1e-5, prob.max() - 1e-5) # 0.5 = p_threshold
    pred = prob > p_thr
    return prob, pred

def ccgmm_pred(loss, targets, thr, reshape = False):
    if reshape:
        loss = loss.reshape(-1, 1)
    num_classes = int(torch.max(targets).item()+1)
    prob = np.zeros(loss.size()[0])
    for c in range(num_classes):
        mask = (targets == c).cpu().numpy()
        gmm = GaussianMixture(n_components=2, max_iter=200, tol=1e-2, reg_covar=5e-4)
        gmm.fit(loss[:,0][mask].reshape(-1,1))

        clean_idx, noisy_idx = gmm.means_.argmin(), gmm.means_.argmax()

        p = gmm.predict_proba(loss[:,0][mask].reshape(-1,1))
        prob[mask] = p[:, clean_idx]

    p_thr = np.clip(thr, prob.min() + 1e-5, prob.max() - 1e-5) # 0.5 = p_threshold
    pred = prob > p_thr
    return prob, pred

def or_ccgmm(losses, targets, thr, reshape = True):
    """
        The probability after the logical OR should be either the mean of the probabilities or the highest, or the mean of all the elements that were correct, i.e. that predicted the same population than the composition.
    """
    probs, preds = np.asarray(list(zip(*np.asarray([ccgmm_pred(losses[:, i], targets, thr, reshape=reshape) for i in range(losses.size(-1))]))))
    pred = np.any(preds, axis=0)
    prob = np.mean(probs, axis=0)
    return prob, pred

def and_ccgmm(losses, targets, thr, reshape = True):
    """
        The probability after the logical AND should be either the mean of the probabilities or the lowest, or the mean of all the elements that were correct, i.e. that predicted the same population than the composition.
    """
    probs, preds = np.asarray(list(zip(*np.asarray([ccgmm_pred(losses[:, i], targets, thr, reshape=reshape) for i in range(losses.size(-1))]))))
    pred = np.all(preds, axis=0)
    prob = np.mean(probs, axis=0)
    return prob, pred

def mean_ccgmm(losses, targets, thr, reshape = True):
    probs, _ = np.asarray(list(zip(*np.asarray([ccgmm_pred(losses[:, i], targets, thr, reshape=reshape) for i in range(losses.size(-1))]))))
    prob = np.mean(probs, axis=0)
    p_thr = np.clip(thr, prob.min() + 1e-5, prob.max() - 1e-5) # 0.5 = p_threshold
    return prob, prob > p_thr

def benchmark(pred, clean_indices):
    tp_tn = clean_indices == pred
    fp_fn = clean_indices != pred
    tp = np.logical_and(tp_tn, clean_indices)
    tn = np.logical_and(tp_tn, ~clean_indices)
    fp = np.logical_and(fp_fn, ~clean_indices)
    fn = np.logical_and(fp_fn, clean_indices)
    precision = tp.sum()/(fp.sum()+fp.sum())
    recall = tp.sum()/(tp.sum()+fn.sum())
    f1_score = recall*precision/(precision+recall)
    accuracy = (clean_indices == pred).sum()/50000
    return {'Acc':accuracy,'F1' :f1_score, '%FP': fp.sum()/50000, 'tp':tp, 'fn':tn, 'fp':fp, 'fn':fn}