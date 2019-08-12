import numpy as np
import torch


def rle2mask(rle, img_shape):
    width = img_shape[0]
    height = img_shape[1]

    mask = np.zeros(width * height).astype(np.uint8)
    if not isinstance(rle, str):
        return np.flipud(np.rot90(mask.reshape(height, width), k=1))
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        mask[int(start):int(start + lengths[index])] = 1
        current_position += lengths[index]

    return np.flipud(np.rot90(mask.reshape(height, width), k=1))


def mask2rle(img):
    tmp = np.rot90(np.flipud(img), k=3)
    rle = []
    last_color = 0;
    start_pos = 0
    end_pos = 0

    tmp = tmp.reshape(-1, 1)
    for i in range(len(tmp)):
        if (last_color == 0) and tmp[i] > 0:
            start_pos = i
            last_color = 1
        elif (last_color == 1) and (tmp[i] == 0):
            end_pos = i-1
            last_color = 0
            rle.append(str(start_pos) + ' ' + str(end_pos - start_pos + 1))
    return " ".join(rle)


def pred2mask(batch_pred):
    prob_mean = torch.mean(batch_pred) * 1.2
    batch_pred[batch_pred < prob_mean] = 0
    batch_pred[batch_pred >= prob_mean] = 1
    return batch_pred
