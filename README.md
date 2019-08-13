# Steel Defect Detection

Pipeline for Kaggle Competition [Severstal: Steel Defect Detection](https://www.kaggle.com/c/severstal-steel-defect-detection)

For training run: 
```shell script
python train.py \
--model_conf=./params/models/unet1.json \
--train_conf=./params/trains/train_unet_local_1.json
```

For making predictions:
```shell script
python submit.py --model_json=unet1.json --model_path=./logs/UNet_best.dat\
 --data_path=./input/severstal-steel-defect-detection/\
 --csv=./input/severstal-steel-defect-detection/sample_submission.csv\ 
--cuda=True
```

Examples for configuration are shown in `./params/`.

When adding new models into `lib/models.py` make sure that it exists in `configs.py`:
```python
from lib.models import UNet
architectures = {
    'simplecnn': UNet
}
```

## Sucessfull models

- unet_kgl.json `0.85 LB` (trained with conf: `train_unet_kaggle_1.json`)