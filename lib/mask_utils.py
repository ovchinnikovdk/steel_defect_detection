import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import tqdm


def rle2mask(mask_rle, shape=(1600, 256)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (width,height) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    if not isinstance(mask_rle, str) or len(mask_rle) <= 1:
        return np.zeros(shape[0] * shape[1], dtype=np.uint8).reshape(shape).T
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


def pred2mask(batch_pred, threshold=0.5):
    batch_pred[batch_pred < threshold] = 0
    batch_pred[batch_pred >= threshold] = 1
    return batch_pred


def save_mask_image(path, imgs, true_mask, pred_mask):
    path = os.path.join(path, 'imgs')
    if not os.path.exists(path):
        os.mkdir(path)
    print('Saving predicted mask into ' + path)
    palet = [(249, 192, 12), (0, 185, 241), (114, 0, 218), (249, 50, 12)]
    palet_pred = [(245, 187, 5), (0, 180, 236), (109, 0, 213), (244, 45, 7)]
    true_mask = true_mask.cpu().numpy().astype('uint8')
    pred_mask = pred_mask.cpu().numpy().astype('uint8')
    imgs = imgs.permute(0, 2, 3, 1).cpu().numpy()
    print(true_mask.shape, pred_mask.shape, imgs.shape)
    for i in tqdm.tqdm(range(len(imgs)), desc='Saving imgs'):
        fig, ax = plt.subplots(figsize=(15, 15))
        img = imgs[i, :, :, :]
        for ch in range(4):
            contours_true, _ = cv2.findContours(true_mask[i, :, :, ch], cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            contours_pred, _ = cv2.findContours(pred_mask[i, :, :, ch], cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            for j in range(len(contours_true)):
                cv2.polylines(img, contours_true[j], True, palet[ch], 2)
            for j in range(len(contours_pred)):
                cv2.polylines(img, contours_pred[j], True, palet_pred[ch], 2)
        ax.set_title(str(i))
        ax.imshow(img)
        plt.savefig(os.path.join(path, str(i) + '.png'))
