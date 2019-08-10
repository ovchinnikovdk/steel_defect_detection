import torch
import numpy as np


def dice(pred_y, truth_y, eps=1e-6):
    shape = pred_y.shape
    # print(shape)
    channels = []
    for i in range(4):
        p = pred_y[:, i, :, :]
        t = truth_y[:, i, :, :]
        pr_flat = p.view(-1, shape[2] * shape[3])
        tr_flat = t.view(-1, shape[2] * shape[3])
        intersection = torch.sum(pr_flat * tr_flat, 1)
        dice_scores = torch.tensor(1.) - 2. * (intersection + eps) / (torch.sum(pr_flat, 1) + torch.sum(tr_flat, 1) + eps)
        channels.append(torch.mean(dice_scores).item())
    # print(dice_scores)
    # return torch.mean(dice_scores).item()
    return np.mean(channels)


# if __name__ == '__main__':
#     a = torch.zeros((20, 4, 200, 300))
#     b = torch.zeros((20, 4, 200, 300))
#     a[:, :, 10:20, :] = 1
#     b[:, :, 2:25, :] = 1
#     print(dice(b, a))