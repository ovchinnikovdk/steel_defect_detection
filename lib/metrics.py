import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score


def dice(pred_y, truth_y, eps=1.):
    shape = pred_y.shape
    dices = []
    for i in range(shape[1]):
        pr_flat = pred_y[:, i, :, :].view(-1, shape[2] * shape[3])
        tr_flat = truth_y[:, i, :, :].view(-1, shape[2] * shape[3])
        intersection = torch.sum(pr_flat * tr_flat, 1)
        dice_scores = (2. * intersection + eps) / (torch.sum(pr_flat, 1) + torch.sum(tr_flat, 1) + eps)
        dices.append(dice_scores)
    dices = torch.cat(dices)
    return torch.mean(dices).item()


def roc_auc(pred_y, truth_y):
    pred_y = pred_y.view(-1).cpu().numpy()
    truth_y = truth_y.view(-1).cpu().numpy()
    return roc_auc_score(truth_y, pred_y)


def accuracy(pred_y, truth_y):
    pred_y = pred_y.view(-1).cpu().numpy()
    truth_y = truth_y.view(-1).cpu().numpy()
    return accuracy_score(truth_y, pred_y)


# if __name__ == '__main__':
#     a = torch.zeros((20, 4, 200, 300))
#     b = torch.zeros((20, 4, 200, 300))
#     a[:, :, 10:20, :] = 1
#     b[:, :, 2:25, :] = 1
#     print(dice(b, a))