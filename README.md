# Audio Classification for ICBHI Respiratory Sounds

A complete PyTorch-based audio classification system for respiratory sound analysis using the ICBHI (International Conference on Biomedical Health Informatics) dataset. Optimized for RTX 3050 4GB with fast training and inference.

## 🚀 Features

- **Efficient Deep Learning Models**: Lightweight CNN and compact ResNet18 architectures
- **Mixed Precision Training**: FP16 training for faster performance on RTX 3050
- **Comprehensive CLI**:  Easy-to-use command-line interface for inference
- **Advanced Preprocessing**:  Mel-spectrogram conversion with audio augmentation
- **Complete Training Pipeline**: TensorBoard logging, early stopping, checkpointing
- **Detailed Validation**:  Accuracy, precision, recall, F1-score, confusion matrices, ROC curves

## 📋 Requirements

- **Hardware**: Intel i5 12500H, RTX 3050 4GB, 16GB DDR4
- **Python**: 3.9+
- **CUDA**:  Compatible version for PyTorch 2.0+

## 🔧 Installation

1. **Install uv** (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. **Clone the repository**:
```bash
git clone https://github.com/AkZuza/audio-classification-icbhi.git
cd audio-classification-icbhi
```

3. **Install dependencies**:
```bash
uv pip install -e .
```

## 📊 Dataset Preparation

1. **Download the ICBHI 2017 Challenge Dataset**:
   - Visit:  https://bhichallenge.med.auth.gr/
   - Download the respiratory sound database

2. **Organize the dataset**:
```
data/ICBHI/
├── audio_and_txt_files/
│   ├── 101_1b1_Al_sc_Meditron. wav
│   ├── 101_1b1_Al_sc_Meditron. txt
│   └── ...
└── ICBHI_final_database/
    └── ... 
```

3. **Expected format**:
   - Audio files: `.wav` format
   - Annotations: `.txt` files with respiratory cycle information
   - Classes: normal, crackles, wheezes, both

## 🏋️ Training

### Basic Training
```bash
python train.py
```

### Custom Configuration
```bash
python train.py --config custom_config.yaml
```

### Training with specific model
```bash
python train.py --model cnn --epochs 50 --batch-size 64
```

### Arguments: 
- `--config`: Path to config file (default: `config.yaml`)
- `--model`: Model architecture (`cnn` or `resnet`, default: `cnn`)
- `--epochs`: Number of training epochs
- `--batch-size`: Batch size for training
- `--learning-rate`: Initial learning rate
- `--device`: Device to use (`cuda` or `cpu`)

## 📈 Validation

### Validate trained model
```bash
python validate.py --model checkpoints/best_model.pt
```

### Validation with custom dataset split
```bash
python validate.py --model checkpoints/best_model.pt --split test
```

This will generate:
- Accuracy, precision, recall, F1-score
- Confusion matrix visualization
- Per-class performance metrics
- ROC curves and AUC scores

## 🎯 CLI Inference

### Classify a single audio file
```bash
python cli.py classify --audio path/to/audio.wav --model checkpoints/best_model.pt
```

### Batch classification
```bash
python cli.py classify-batch --input-dir path/to/audio/files --model checkpoints/best_model. pt --output results.csv
```

### Show model information
```bash
python cli.py info --model checkpoints/best_model.pt
```

### CLI Options: 
- `--audio`: Path to audio file (for single classification)
- `--input-dir`: Directory containing audio files (for batch)
- `--model`: Path to trained model checkpoint
- `--output`: Output file for results (CSV or JSON)
- `--device`: Device to use (`cuda` or `cpu`)

## ⚙️ Configuration

Edit `config.yaml` to customize training: 

```yaml
data:
  dataset_path: "data/ICBHI"
  sample_rate: 16000
  n_mels: 128
  duration: 5.0
  augmentation: true

model:
  architecture: "cnn"  # or "resnet"
  num_classes: 4
  dropout: 0.3

training:
  batch_size: 32
  epochs:  100
  learning_rate:  0.001
  mixed_precision: true
  gradient_accumulation_steps: 2
  early_stopping_patience: 15
```

## 🏗️ Model Architectures

### 1. Lightweight CNN
- 5 convolutional blocks with batch normalization
- Global average pooling
- Optimized for 4GB VRAM
- Fast inference (~10ms per sample)

### 2. Compact ResNet18
- Modified ResNet18 for audio spectrograms
- Reduced channels for memory efficiency
- Optional ImageNet pretraining
- Higher accuracy, slightly slower

## 📊 Performance

Optimizations for RTX 3050 4GB:
- ✅ Mixed precision (FP16) training
- ✅ Gradient accumulation
- ✅ Efficient data loading
- ✅ Memory-optimized architectures
- ✅ Batch inference

Expected performance:
- **Training time**: ~2-3 hours for 100 epochs
- **Inference time**: ~10-20ms per sample
- **Memory usage**: <3. 5GB VRAM
- **Validation accuracy**: 70-85% (depends on dataset split)

## 📁 Project Structure

```
audio-classification-icbhi/
├── src/
│   ├── models/          # Model architectures
│   ├── data/            # Data processing
│   ├── training/        # Training logic
│   └── utils/           # Utilities
├── train.py             # Training script
├── validate.py          # Validation script
├── cli.py               # CLI inference tool
├── config.yaml          # Configuration
├── pyproject.toml       # Dependencies
└── README.md            # Documentation
```

## 🐛 Troubleshooting

### CUDA Out of Memory
- Reduce `batch_size` in `config.yaml`
- Increase `gradient_accumulation_steps`
- Use `mixed_precision: true`

### Slow Training
- Enable `mixed_precision: true`
- Increase `num_workers` for data loading
- Ensure `pin_memory: true`

### Audio Loading Errors
- Check audio file format (should be . wav)
- Verify sample rate matches config
- Ensure proper dataset directory structure

## 📝 Citation

If you use the ICBHI dataset, please cite:
```
@inproceedings{icbhi2017,
  title={ICBHI 2017 Challenge respiratory sound database},
  author={Rocha, Bruno M and others},
  booktitle={International Conference on Biomedical Health Informatics},
  year={2017}
}
```

## 📄 License

MIT License - feel free to use for research and commercial purposes. 

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request. 
