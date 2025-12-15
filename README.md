# Math Mind: GRPO Training for Mathematical Reasoning

This repository implements Group Relative Policy Optimization (GRPO) for training language models on mathematical reasoning tasks, specifically using the GSM8K dataset. 

## Overview

This project fine-tunes language models (e.g., Qwen2.5-3B-Instruct) using GRPO to improve their mathematical reasoning capabilities. The training uses reward functions that evaluate:

1. **Format correctness**: Checks if the model follows the structured reasoning format
2. **Answer correctness**: Validates if the final numerical answer is correct
3. **Stepwise reasoning**: Evaluates the correctness and progression of intermediate steps

## Project Structure

```
math-mind/
├── src/
│   ├── config.py              # Configuration constants and hyperparameters
│   ├── data/
│   │   ├── __init__.py
│   │   └── processing.py      # Dataset loading and preprocessing
│   ├── rewards/
│   │   ├── __init__.py
│   │   ├── format_rewards.py  # Format checking rewards
│   │   ├── correctness_rewards.py  # Answer correctness rewards
│   │   ├── stepwise_rewards.py     # Stepwise reasoning rewards
│   │   ├── equation_parser.py      # Equation parsing utilities
│   │   └── combined_rewards.py     # Combined reward function
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py         # Model setup and training logic
│   └── inference/
│       ├── __init__.py
│       └── inference.py       # Inference and evaluation utilities
├── train.py                   # Main training script
├── inference.py               # Main inference script
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd math-mind
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

**Note**: The requirements include PyTorch with CUDA support. Adjust the PyTorch installation in `requirements.txt` based on your CUDA version. For CPU-only systems, use:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Optional dependencies**:
- `flash-attn`: For faster training (requires compatible CUDA setup)
- `vllm`: For faster generation in distributed training

### 4. Setup HuggingFace authentication

You'll need a HuggingFace token to download models and datasets:

```bash
huggingface-cli login
```

Or set the `HF_TOKEN` environment variable:

```bash
export HF_TOKEN=your_token_here
```

### 5. Setup Wandb (optional, for experiment tracking)

```bash
wandb login
```

## Usage

### Training

Train a model using GRPO on the GSM8K dataset:

```bash
python train.py \
    --model-name "Qwen/Qwen2.5-3B-Instruct" \
    --output-dir "qwen25-3b-grpo-onlyfinalreward-scaling1" \
    --hub-model-id "your-username/qwen25-3b-grpo-onlyfinalreward-scaling1" \
    --train-limit 500 \
    --eval-limit 50 \
    --wandb-entity "your-wandb-entity"
```

**Arguments**:
- `--model-name`: HuggingFace model identifier (default: Qwen/Qwen2.5-3B-Instruct)
- `--output-dir`: Output directory for checkpoints and final model (required)
- `--hub-model-id`: Optional HuggingFace Hub model ID for pushing checkpoints
- `--dataset-name`: Dataset name (default: openai/gsm8k)
- `--train-limit`: Limit number of training examples (optional, for quick testing)
- `--eval-limit`: Limit number of evaluation examples (default: 50)
- `--wandb-entity`: Wandb entity/username (optional)
- `--run-name`: Run name for wandb (optional)

**Example with minimal settings**:

```bash
python train.py --output-dir "my-grpo-model" --eval-limit 20
```

### Inference

Run inference on a trained model:

```bash
python inference.py \
    --model-id "your-username/qwen25-3b-grpo-onlyfinalreward-scaling1" \
    --output-file "outputs/test_results.json" \
    --limit 100 \
    --max-new-tokens 1024
```

**Arguments**:
- `--model-id`: HuggingFace model ID or local path (required)
- `--output-file`: Path to save inference results JSON file (required)
- `--dataset-name`: Dataset name (default: openai/gsm8k)
- `--split`: Dataset split to use (default: test)
- `--limit`: Limit number of examples to process (optional)
- `--max-new-tokens`: Maximum number of new tokens to generate (default: 1024)

**Example**:

```bash
python inference.py \
    --model-id "thegrey07/qwen25-3b-grpo-onlyfinalreward-scaling1" \
    --output-file "outputs/my_results.json" \
    --limit 50
```

## Configuration

Training hyperparameters and other settings can be modified in `src/config.py`:

- **Format tokens**: `REASONING_START`, `REASONING_END`, `ANSWER_SEPARATOR`, `EQUATION_SEPARATOR`
- **System prompt**: `SYSTEM_PROMPT`
- **Training config**: `TRAINING_CONFIG` (learning rate, batch size, max steps, etc.)
- **LoRA config**: `LORA_CONFIG` (rank, alpha, dropout, target modules)
- **Reward scaling**: `SCALING_FACTOR`

## Reward Functions

The repository includes several reward functions in `src/rewards/`:

1. **`check_format_rewards`**: Validates structured reasoning format
2. **`check_answer_correctness`**: Checks final numerical answer correctness
3. **`reward_stepwise`**: Evaluates stepwise reasoning quality
4. **`combined_reward_fn`**: Combines all reward components

You can modify `train.py` to use different reward functions:

```python
from src.rewards import combined_reward_fn, check_answer_correctness

# Use only final answer reward (current default)
reward_funcs=[check_answer_correctness]

# Or use combined rewards
reward_funcs=[combined_reward_fn]
```

## Output Format

The model is trained to produce structured reasoning outputs:

```
<begin_steps>
<STEP 1>
Explanation...
@@@@ 10 * 7 = 70 @@@@
</STEP>

<STEP 2>
Explanation...
@@@@ 70 / 2 = 35 @@@@
</STEP>
</end_steps>

#### 35
```

## Hardware Requirements

- **GPU**: Recommended (CUDA-capable GPU with at least 16GB VRAM for 3B models)
- **Memory**: 32GB+ RAM recommended
- **Storage**: ~10GB for models and checkpoints

Training on CPU is possible but very slow and not recommended.

## Troubleshooting

### Out of Memory Errors

- Reduce `per_device_train_batch_size` in `src/config.py`
- Reduce `max_completion_length` or `max_prompt_length`
- Use smaller models (e.g., Qwen2.5-0.5B-Instruct)
- Enable gradient checkpointing (already enabled by default)

### Installation Issues

- Ensure your CUDA version matches PyTorch requirements
- For flash-attn, you may need to install from source if pre-built wheels aren't available
- Some dependencies may have platform-specific requirements

### Training Instability

- Adjust `beta` (KL coefficient) in `TRAINING_CONFIG`
- Reduce learning rate
- Adjust reward scaling factor

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Acknowledgments

- Based on GRPO (Group Relative Policy Optimization) from the DeepSeekMath paper
- Uses TRL (Transformers Reinforcement Learning) library from HuggingFace
- Trained on GSM8K dataset from OpenAI

## References

- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
- [TRL Documentation](https://huggingface.co/docs/trl)
- [GSM8K Dataset](https://huggingface.co/datasets/openai/gsm8k)

