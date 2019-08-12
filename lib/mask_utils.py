import numpy as np
import torch


# def rle2mask(rle, img_shape):
#     width = img_shape[0]
#     height = img_shape[1]
#
#     mask = np.zeros(width * height).astype(np.uint8)
#     if not isinstance(rle, str):
#         return np.flipud(np.rot90(mask.reshape(height, width), k=1))
#     array = np.asarray([int(x) for x in rle.split()])
#     starts = array[0::2]
#     lengths = array[1::2]
#
#     current_position = 0
#     for index, start in enumerate(starts):
#         mask[int(start):int(start + lengths[index])] = 1
#         current_position += lengths[index]
#
#     return np.flipud(np.rot90(mask.reshape(height, width), k=1))


# def mask2rle(img):
#     tmp = np.rot90(np.flipud(img), k=3)
#     rle = []
#     last_color = 0;
#     start_pos = 0
#     end_pos = 0
#
#     tmp = tmp.reshape(-1, 1)
#     for i in range(len(tmp)):
#         if (last_color == 0) and tmp[i] > 0:
#             start_pos = i
#             last_color = 1
#         elif (last_color == 1) and (tmp[i] == 0):
#             end_pos = i-1
#             last_color = 0
#             rle.append(str(start_pos) + ' ' + str(end_pos - start_pos + 1))
#     return " ".join(rle)

def rle2mask(mask_rle, shape=(1600,256)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (width,height) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T


def mask2rle(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def pred2mask(batch_pred):
    prob_mean = torch.mean(batch_pred) * 1.2
    batch_pred[batch_pred < prob_mean] = 0
    batch_pred[batch_pred >= prob_mean] = 1
    return batch_pred
