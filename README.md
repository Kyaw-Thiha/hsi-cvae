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
- Transformer Repeat-Z: `--config config/models/transformer_repeatz.yaml`
- Conformer: `--config config/models/conformer.yaml`

### 🔍 Predict / Sampling
Predict commands automatically include `config/base.yaml` and `config/predict.yaml`. Append additional configs to pick the model and custom predict grids:
```bash
# Transformer model + half-sum predict grid
python main.py predict --config config/models/transformer.yaml --config config/predict/half.yaml --ckpt_path outputs/run/checkpoints/best/transformer.ckpt

# Conformer model + training-grid predict config
python main.py predict --config config/models/conformer.yaml --config config/predict/training.yaml --ckpt_path outputs/run/checkpoints/best/conformer.ckpt
```

### 📈 Visualize Original Dataset
```bash
python -m callbacks.original_line_charts
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

---
## Configs
### MLP Configs
- config/models/mlp/mlp_5m.yaml (depth 10, ~5.02M params)
- config/models/mlp/mlp_10m.yaml (depth 19, ~9.76M params)
- config/models/mlp/mlp_20m.yaml (depth 38, ~19.78M params)
- config/models/mlp/mlp_40m.yaml (depth 76, ~39.82M params)
- config/models/mlp/mlp_80m.yaml (depth 153, ~80.43M params)

### CNN Configs
- config/models/cnn/cnn_5m.yaml (width 128, depth 24, ~4.97M)
- config/models/cnn/cnn_10m.yaml (width 192, depth 28, ~10.04M)
- config/models/cnn/cnn_20m.yaml (width 256, depth 38, ~19.99M)
- config/models/cnn/cnn_40m.yaml (width 384, depth 37, ~40.00M)
- config/models/cnn/cnn_80m.yaml (width 512, depth 45, ~80.10M)
