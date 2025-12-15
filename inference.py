#!/usr/bin/env python3
"""Main inference script for evaluating trained models."""

import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.data import load_and_process_dataset
from src.inference import run_inference


def main():
    parser = argparse.ArgumentParser(description="Run inference on a trained model")
    parser.add_argument(
        "--model-id",
        type=str,
        required=True,
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path to save inference results JSON file",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="openai/gsm8k",
        help="Dataset name to load",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to use",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples to process",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Maximum number of new tokens to generate",
    )
    
    args = parser.parse_args()

    # Load dataset
    print(f"📚 Loading {args.dataset_name} ({args.split})...")
    dataset = load_and_process_dataset(
        dataset_name=args.dataset_name,
        split=args.split,
        limit=args.limit,
    )

    # Load model and tokenizer
    print(f"🔄 Loading model: {args.model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=True,
    )

    # Run inference
    print("🚀 Running inference...")
    run_inference(
        dataset=dataset,
        model=model,
        tokenizer=tokenizer,
        output_file=args.output_file,
        max_new_tokens=args.max_new_tokens,
    )
    
    print("✅ Inference complete!")


if __name__ == "__main__":
    main()

