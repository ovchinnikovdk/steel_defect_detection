# Steal Defect Detection

Pipeline for Kaggle Competition [Severstal: Steel Defect Detection](https://www.kaggle.com/c/severstal-steel-defect-detection)

For training run: 
```shell script
python train.py \
--model_conf=./params/models/simple_cnn.json \
--train_conf=./params/trains/train_conf_1.json
```

For making predictions:
```shell script
python submit.py --model_json=./params/models/unet.json\
 --model_path=./logs/UNet_best.dat\
 --data_path=./input/severstal-steel-defect-detection/\
 --csv=./input/severstal-steel-defect-detection/sample_submission.csv\ 
--cuda=True
```

Examples for configuration are shown in `./params/`.

When adding new models into `lib/models.py` make sure that it exists in `configs.py`:
```python
from lib.models import SimpleCNN
architectures = {
    'simplecnn': SimpleCNN
}
```