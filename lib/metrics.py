import torch


def dice(pred_y, truth_y, eps=1.):
    shape = pred_y.shape
    dices = torch.Tensor()
    for i in range(shape[1]):
        pr_flat = pred_y[:, i, :, :].view(-1, shape[2] * shape[3])
        tr_flat = truth_y[:, i, :, :].view(-1, shape[2] * shape[3])
        intersection = torch.sum(pr_flat * tr_flat, 1)
        dice_scores = 2. * (intersection + eps) / (torch.sum(pr_flat, 1) + torch.sum(tr_flat, 1) + eps)
        dices = torch.cat((dices, dice_scores))
    return torch.mean(dices).item()


# if __name__ == '__main__':
#     a = torch.zeros((20, 4, 200, 300))
#     b = torch.zeros((20, 4, 200, 300))
#     a[:, :, 10:20, :] = 1
#     b[:, :, 2:25, :] = 1
#     print(dice(b, a))