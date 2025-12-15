# Quick Start Guide

This guide will help you get started with training and inference quickly.

## Prerequisites

1. Python 3.8+
2. CUDA-capable GPU (recommended)
3. HuggingFace account and token
4. Wandb account (optional, for experiment tracking)

## Setup (5 minutes)

```bash
# 1. Clone and navigate to repository
cd math-mind

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Login to HuggingFace
huggingface-cli login

# 5. (Optional) Login to Wandb
wandb login
```

## Quick Training Example

Train on a small subset for testing:

```bash
python train.py \
    --output-dir "test-model" \
    --train-limit 100 \
    --eval-limit 20
```

## Quick Inference Example

Run inference on a pretrained model:

```bash
python inference.py \
    --model-id "thegrey07/qwen25-3b-grpo-onlyfinalreward-scaling1" \
    --output-file "outputs/test_results.json" \
    --limit 10
```

## Full Training Example

Train with full configuration:

```bash
python train.py \
    --model-name "Qwen/Qwen2.5-3B-Instruct" \
    --output-dir "qwen25-3b-grpo-onlyfinalreward-scaling1" \
    --hub-model-id "your-username/qwen25-3b-grpo-onlyfinalreward-scaling1" \
    --train-limit 500 \
    --eval-limit 50 \
    --wandb-entity "your-wandb-entity"
```

## Common Issues

**Out of Memory?**
- Reduce `--train-limit` or remove it to use full dataset
- Use a smaller model like `Qwen/Qwen2.5-0.5B-Instruct`
- Modify batch size in `src/config.py`

**Import Errors?**
- Make sure you're running from the repository root directory
- Check that virtual environment is activated
- Verify all dependencies are installed: `pip install -r requirements.txt`

**Model Download Issues?**
- Ensure you're logged in: `huggingface-cli login`
- Check your internet connection
- Some models require accepting terms on HuggingFace website first

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Explore `src/config.py` to customize hyperparameters
- Modify reward functions in `src/rewards/` to experiment with different reward strategies

