"""Inference functions for generating and evaluating model outputs."""

import json
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.config import ANSWER_SEPARATOR, INFERENCE_CONFIG


def generate_output(prompt_messages, tokenizer, model, max_new_tokens=None):
    """
    Generate output from model given prompt messages.
    
    Args:
        prompt_messages: list of {"role": ..., "content": ...}
        tokenizer: Tokenizer instance
        model: Model instance
        max_new_tokens: Maximum number of new tokens to generate
    
    Returns:
        full_text: Full generated text
        final_ans: Extracted final answer
    """
    if max_new_tokens is None:
        max_new_tokens = INFERENCE_CONFIG["max_new_tokens"]
    
    # Convert conversation format to chat template
    text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    full_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Split on the exact token "assistant\n"
    split_key = "assistant\n"

    if split_key in full_text:
        # Take everything after "assistant\n"
        full_text = full_text.split(split_key, 1)[1].strip()
    else:
        full_text = ""

    if ANSWER_SEPARATOR in full_text:
        # Take everything after answer separator
        final_ans = full_text.split(ANSWER_SEPARATOR, 1)[1].strip()
    else:
        final_ans = ""

    return full_text, final_ans


def run_inference(dataset, model, tokenizer, output_file, max_new_tokens=None):
    """
    Run inference on a dataset and save results.
    
    Args:
        dataset: Dataset to run inference on
        model: Model instance
        tokenizer: Tokenizer instance
        output_file: Path to save results JSON file
        max_new_tokens: Maximum number of new tokens to generate
    """
    results = []
    print("🚀 Running inference...")

    for i, example in enumerate(tqdm(dataset)):
        prompt = example["prompt"]  # conversation messages
        completion, final_ans = generate_output(prompt, tokenizer, model, max_new_tokens)

        results.append({
            "prompt": prompt,
            "completion": completion,
            "answer": example["answer"],
            "model_answer": final_ans,
            "correct": example["answer"] == final_ans
        })
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1} samples")
            # Save intermediate results
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)

    # Save final results
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Saved {len(results)} model outputs to {output_file}")
    
    # Print accuracy
    correct = sum(1 for r in results if r["correct"])
    accuracy = correct / len(results) * 100
    print(f"📊 Accuracy: {correct}/{len(results)} ({accuracy:.2f}%)")

