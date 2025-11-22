# HSI-CVAE
This is a Conditional Variational AutoEncoder for generating synthetic hyperspectral data, developed for usage
by FINCH satellite of University of Toronto Aerospace Team.

## Training the Model
### 🔧 Training
```bash
python main.py fit --config config/cvae.yaml
```

### 🔧 Training from a checkpoint
```bash
python main.py fit --config config/cvae.yaml --ckpt_path runs/train_/checkpoints/interval/hsdt-epoch10.ckpt
```

### 🔧 Smoke Test
```bash
python main.py fit --config config/cvae.yaml --trainer.profiler=null --trainer.fast_dev_run=True
```

### 🔧 Best batch finder
```bash
python main.py fit --config config/cvae.yaml --run_batch_size_finder true --batch_size_finder_mode power
```

### 🔧 Best learning rate finder
```bash
python main.py fit --config config/cvae.yaml --run_lr_finder true 
```
