import argparse
from lib.configs import ConfigFactory
from lib.dataset import *
from lib.submission_utils import *
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, help='JSON File Submission Configuration')

    args = parser.parse_args()
    configurator = ConfigFactory()
    models, df, data_path, cuda = configurator.build_submit_env(args.config_path)
    result_df = None
    if 'prediction' in models:
        predictor = models['prediction']
        df['filename'] = df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])
        dataloader = DataLoader(dataset=SteelPredictionDataset(data_path, df, subset='test'), batch_size=80)
        clean, df = CleanSteelPredictor(df, predictor, dataloader, cuda).call()
        result_df = clean
    if 'segmentation' in models:
        segmentation = models['segmentation']
        df['filename'] = df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])
        df['class'] = df['ImageId_ClassId'].apply(lambda x: int(x.split('_')[1]))
        dataloader = DataLoader(dataset=SteelDataset(data_path, df, subset='test'), batch_size=80)
        df = SteelSegmentation(df, segmentation, dataloader, cuda).call()
        if result_df is not None:
            print(df.head(10))
            result_df = result_df.append(df).reset_index()
        else:
            result_df = df

    test_df = result_df[['ImageId_ClassId', 'EncodedPixels']]
    test_df.head(10)
    test_df.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    main()

#python submit.py --model_json=./params/models/unet1.json --model_path=./logs/UNet_best.dat --data_path=./input/severstal-st
#eel-defect-detection/ --csv=./input/severstal-steel-defect-detection/sample_submission.csv --cuda=True

