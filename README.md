# Steel Defect Detection

Pipeline for Kaggle Competition [Severstal: Steel Defect Detection](https://www.kaggle.com/c/severstal-steel-defect-detection)

For training run: 
```shell script
python train.py \
--model_conf=./params/models/unet1.json \
--train_conf=train_local_1.json
```
Or:
```shell script
python train.py --model_conf=./params/models/pspnet_v1.json --train_conf=./params/trains/train_local_2.json
```

For making predictions:
```shell script
python submit.py --config_path=./params/submit/2_model_submission.json
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