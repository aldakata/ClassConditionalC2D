import numpy as np
from sklearn.mixture import GaussianMixture
import torch

def log_loss(epoch, losses_clean, pred, loss_log):
    loss_log.write('{},{},{},{},{}\n'.format(epoch, losses_clean[pred].mean(dim=1), losses_clean[pred].std(dim=1),
                                                        losses_clean[~pred].mean(dim=1), losses_clean[~pred].std(dim=1)))
    loss_log.flush()

def gmm_pred(loss, reshape = False):
    if reshape:
        loss = loss.reshape(-1,1)
    gmm = GaussianMixture(n_components=2, max_iter=200, tol=1e-2, reg_covar=5e-4)
    gmm.fit(loss)

    clean_idx, noisy_idx = gmm.means_.argmin(), gmm.means_.argmax()

    prob = gmm.predict_proba(loss)
    prob = prob[:, clean_idx]

    p_thr = np.clip(0.5, prob.min() + 1e-5, prob.max() - 1e-5) # 0.5 = p_threshold
    pred = prob > p_thr
    return gmm, clean_idx, noisy_idx, pred, prob

def gmm_pred_class_dependant(loss, targets, reshape = False):
    num_classes = int(torch.max(targets).item()+1)
    print(num_classes)
    prob = np.zeros(loss.size()[0])
    for c in range(num_classes):
        mask = (targets == c).cpu().numpy()
        gmm = GaussianMixture(n_components=2, max_iter=200, tol=1e-2, reg_covar=5e-4)
        gmm.fit(loss[:,0][mask].reshape(-1,1))

        clean_idx, noisy_idx = gmm.means_.argmin(), gmm.means_.argmax()

        p = gmm.predict_proba(loss[:,0][mask].reshape(-1,1))
        prob[mask] = p[:, clean_idx]

    p_thr = np.clip(0.5, prob.min() + 1e-5, prob.max() - 1e-5) # 0.5 = p_threshold
    pred = prob > p_thr
    return gmm, clean_idx, noisy_idx, pred, prob
