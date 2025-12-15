#!/usr/bin/env python3
"""Main training script for GRPO mathematical reasoning."""

import argparse
import os
from src.data import load_and_process_dataset
from src.rewards import check_answer_correctness
from src.training import setup_model_and_trainer, train_model
from src.config import TRAINING_CONFIG


def main():
    parser = argparse.ArgumentParser(description="Train a model using GRPO on GSM8K")
    parser.add_argument(
        "--model-name",
        type=str,
        default=TRAINING_CONFIG["model_name"],
        help="HuggingFace model identifier",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for checkpoints and final model",
    )
    parser.add_argument(
        "--hub-model-id",
        type=str,
        default=None,
        help="HuggingFace Hub model ID (optional, for pushing to hub)",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="openai/gsm8k",
        help="Dataset name to load",
    )
    parser.add_argument(
        "--train-limit",
        type=int,
        default=None,
        help="Limit number of training examples (for quick testing)",
    )
    parser.add_argument(
        "--eval-limit",
        type=int,
        default=50,
        help="Limit number of evaluation examples",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Wandb entity/username (optional)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Run name for wandb (optional)",
    )
    
    args = parser.parse_args()

    # Load datasets
    print("📚 Loading datasets...")
    train_dataset = load_and_process_dataset(
        dataset_name=args.dataset_name,
        split="train",
        limit=args.train_limit,
    )
    
    eval_dataset = load_and_process_dataset(
        dataset_name=args.dataset_name,
        split="test",
        limit=args.eval_limit,
    )

    # Setup model and trainer
    print("🔧 Setting up model and trainer...")
    trainer, tokenizer = setup_model_and_trainer(
        model_name=args.model_name,
        reward_funcs=[check_answer_correctness],  # Use only final answer reward
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=args.output_dir,
        hub_model_id=args.hub_model_id,
        use_wandb=True,
        wandb_entity=args.wandb_entity,
    )

    # Train
    print("🚀 Starting training...")
    train_model(
        trainer=trainer,
        output_dir=args.output_dir,
        run_name=args.run_name or args.output_dir,
        wandb_entity=args.wandb_entity,
    )
    
    print("✅ Training complete!")


if __name__ == "__main__":
    main()

