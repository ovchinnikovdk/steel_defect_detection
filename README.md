# Steal Defect Detection

For training run: 
```shell script
python train.py \
--model_conf=./params/models/simple_cnn.json \
--train_conf=./params/trains/train_conf_1.json
```

Examples for configuration are shown in `./params/`.

When adding new models into `lib/models.py` make sure that it exists in `configs.py`:
```python
from lib.models import SimpleCNN
architectures = {
    'simplecnn': SimpleCNN
}
```