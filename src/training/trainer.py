"""Model and trainer setup for GRPO."""

import torch
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
from transformers.trainer_utils import get_last_checkpoint

from src.config import (
    TRAINING_CONFIG,
    LORA_CONFIG,
    USE_4BIT_QUANTIZATION,
)


def setup_model_and_trainer(
    model_name,
    reward_funcs,
    train_dataset,
    eval_dataset,
    output_dir,
    hub_model_id=None,
    use_wandb=True,
    wandb_project="huggingface",
    wandb_entity=None,
):
    """
    Setup model, tokenizer, and GRPO trainer.
    
    Args:
        model_name: HuggingFace model identifier
        reward_funcs: List of reward functions
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        output_dir: Output directory for checkpoints
        hub_model_id: Optional HuggingFace Hub model ID
        use_wandb: Whether to use wandb logging
        wandb_project: Wandb project name
        wandb_entity: Wandb entity/username
    
    Returns:
        trainer: GRPOTrainer instance
        tokenizer: Tokenizer instance
    """
    # Setup quantization if enabled
    bnb_config = None
    if USE_4BIT_QUANTIZATION:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

    # Load model
    print(f"🔄 Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=bnb_config,
        trust_remote_code=True,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Setup LoRA
    peft_config = LoraConfig(
        r=LORA_CONFIG["r"],
        lora_alpha=LORA_CONFIG["lora_alpha"],
        lora_dropout=LORA_CONFIG["lora_dropout"],
        target_modules=LORA_CONFIG["target_modules"],
    )

    # Training arguments
    training_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=TRAINING_CONFIG["learning_rate"],
        lr_scheduler_type=TRAINING_CONFIG["lr_scheduler_type"],
        logging_steps=TRAINING_CONFIG["logging_steps"],
        max_steps=TRAINING_CONFIG["max_steps"],
        per_device_train_batch_size=TRAINING_CONFIG["per_device_train_batch_size"],
        gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation_steps"],
        gradient_checkpointing=True,
        max_grad_norm=TRAINING_CONFIG["max_grad_norm"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=True,
        bf16=False,
        disable_tqdm=False,
        # GRPO specific parameters
        max_prompt_length=TRAINING_CONFIG["max_prompt_length"],
        max_completion_length=TRAINING_CONFIG["max_completion_length"],
        num_generations=TRAINING_CONFIG["num_generations"],
        beta=TRAINING_CONFIG["beta"],
        # Checkpoints + Auto-resume
        save_steps=TRAINING_CONFIG["save_steps"],
        save_total_limit=TRAINING_CONFIG["save_total_limit"],
        report_to="wandb" if use_wandb else None,
    )

    if hub_model_id:
        training_args.push_to_hub = True
        training_args.hub_strategy = "checkpoint"
        training_args.hub_model_id = hub_model_id

    # Create trainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )

    # Initialize optimizer and scheduler
    trainer.create_optimizer_and_scheduler(num_training_steps=TRAINING_CONFIG["max_steps"])
    
    print(f"✅ Model and trainer setup complete")
    print(f"   Total params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    return trainer, tokenizer


def train_model(trainer, output_dir, run_name=None, wandb_entity=None):
    """
    Train the model with checkpoint resuming support.
    
    Args:
        trainer: GRPOTrainer instance
        output_dir: Output directory for checkpoints
        run_name: Optional run name for wandb
        wandb_entity: Optional wandb entity
    """
    import wandb
    from src.config import TRAINING_CONFIG

    # Initialize wandb if not already initialized
    if wandb.run is None and trainer.args.report_to == "wandb":
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        if run_name is None:
            run_name = f"{output_dir}_{timestamp}"
        
        wandb.init(
            project="huggingface",
            entity=wandb_entity,
            name=run_name,
            config={
                "output_dir": output_dir,
                "learning_rate": TRAINING_CONFIG["learning_rate"],
                "batch_size": TRAINING_CONFIG["per_device_train_batch_size"],
                "max_steps": TRAINING_CONFIG["max_steps"],
            }
        )

    # Check for existing checkpoint
    last_ckpt = get_last_checkpoint(output_dir)

    if last_ckpt is not None:
        print(f"🔄 Resuming from checkpoint: {last_ckpt}")
        trainer.train(resume_from_checkpoint=last_ckpt)
    else:
        print("🚀 Starting training from scratch")
        trainer.train()

    # Save final model
    print(f"💾 Saving model to {output_dir}")
    trainer.save_model(output_dir)
    
    if trainer.args.push_to_hub:
        print(f"📤 Pushing model to hub: {trainer.args.hub_model_id}")
        trainer.push_to_hub(dataset_name="openai/gsm8k")

