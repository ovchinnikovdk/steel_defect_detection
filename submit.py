import argparse
from lib.configs import ConfigFactory
from lib.dataset import StealDataset
from lib.mask_utils import pred2mask, mask2rle
from torch.utils.data import DataLoader
import torch
import pandas as pd
import tqdm
import cv2
from joblib import Parallel, delayed
import multiprocessing as mp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_json", type=str, help='JSON File Model Configuration')
    parser.add_argument("--model_path", type=str, help='Trained model path')
    parser.add_argument("--data_path", type=str, help='Data path contained images')
    parser.add_argument("--csv", type=str, help='Csv file with test data')
    parser.add_argument('--cuda', type=bool, help='Is it possible to use CUDA')

    args = parser.parse_args()
    configurator = ConfigFactory()
    model = configurator.build_model(args.model_json)
    model.load_state_dict(torch.load(args.model_path))
    if args.cuda:
        model.cuda()
    else:
        model.cpu()
    df = pd.read_csv(args.csv)[:100]
    df['filename'] = df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])
    df['class'] = df['ImageId_ClassId'].apply(lambda x: int(x.split('_')[1]))
    dataset = StealDataset(args.data_path, df, subset='test')
    data_loader = DataLoader(dataset=dataset, batch_size=80)
    predict(model, df, data_loader, args.cuda)


def predict(model, test_df, test_loader, cuda):
    with torch.no_grad():
        predicts = []
        model.eval()
        for data, label in tqdm.tqdm(test_loader, desc='Predicting batches'):
            if cuda:
                data = data.cuda()
            output = model(data)
            output = output.cpu()
            rles = Parallel(n_jobs=mp.cpu_count())(delayed(post_process)(img, lab) for img, lab in zip(output, label))
            # for img in output:
            #     predict.append(rle)
            predicts += rles

        test_df['EncodedPixels'] = predicts
        test_df = test_df[['ImageId_ClassId', 'EncodedPixels']]
        test_df.head(10)
        test_df.to_csv('submission.csv', index=False)


def post_process(img, lab):
    mask = pred2mask(img[lab-1])
    mask = mask.numpy()
    resized = cv2.resize(mask, (1600, 256))
    return mask2rle(resized)


if __name__ == '__main__':
    main()

#python submit.py --model_json=./params/models/unet1.json --model_path=./logs/UNet_best.dat --data_path=./input/severstal-st
#eel-defect-detection/ --csv=./input/severstal-steel-defect-detection/sample_submission.csv --cuda=True

