# HSI-CVAE
This is a Conditional Variational AutoEncoder for generating synthetic hyperspectral data, developed for usage
by FINCH satellite of University of Toronto Aerospace Team.

## Training the Model
### 🔧 Training
```bash
python main.py fit --config config/models/mlp.yaml
```
`config/base.yaml` loads automatically. Swap the model config for other backends:
- MLP: `--config config/models/mlp.yaml`
- CNN: `--config config/models/cnn.yaml`
- Transformer: `--config config/models/transformer.yaml`
- Conformer: `--config config/models/conformer.yaml`

### 🔍 Predict / Sampling
Predict commands automatically include `config/base.yaml` and `config/predict.yaml`. Append additional configs to pick the model and custom predict grids:
```bash
# Transformer model + half-sum predict grid
python main.py predict --config config/models/transformer.yaml --config config/predict/half.yaml --ckpt_path outputs/run/checkpoints/best/transformer.ckpt

# Conformer model + training-grid predict config
python main.py predict --config config/models/conformer.yaml --config config/predict/training.yaml --ckpt_path outputs/run/checkpoints/best/conformer.ckpt
```

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
