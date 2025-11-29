# HSI-CVAE
This is a Conditional Variational AutoEncoder for generating synthetic hyperspectral data, developed for usage
by FINCH satellite of University of Toronto Aerospace Team.

## Training the Model
### 🔧 Training
```bash
python main.py fit --config config/models/mlp.yaml
```
`config/base.yaml` loads automatically; swap the model config for `config/models/cnn.yaml` or `config/models/transformer.yaml` to train other backends.

### 🔧 Training from a checkpoint
```bash
python main.py fit --config config/models/mlp.yaml --ckpt_path runs/train_/checkpoints/interval/hsdt-epoch10.ckpt
```

### 🔧 Smoke Test
```bash
python main.py fit --config config/models/mlp.yaml --trainer.profiler=null --trainer.fast_dev_run=True
```

### 🔧 Best batch finder
```bash
python main.py fit --config config/models/mlp.yaml --run_batch_size_finder true --batch_size_finder_mode power
```

### 🔧 Best learning rate finder
```bash
python main.py fit --config config/models/mlp.yaml --run_lr_finder true
```
