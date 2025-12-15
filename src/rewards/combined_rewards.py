"""Combined reward function."""

import random
import wandb
from src.rewards.stepwise_rewards import reward_stepwise
from src.rewards.correctness_rewards import check_answer_correctness
from src.rewards.format_rewards import check_format_rewards


def combined_reward_fn(completions, answer, **kwargs):
    """Combine stepwise, answer correctness, and format rewards."""
    step_rewards = reward_stepwise(completions, answer, **kwargs)
    answer_rewards = check_answer_correctness(completions, answer, **kwargs)
    format_rewards = check_format_rewards(completions, answer, **kwargs)

    # Sum components elementwise
    final_reward = [s + a + f for s, a, f in zip(step_rewards, answer_rewards, format_rewards)]

    # Add perturbation if all rewards are equal
    if len(final_reward) > 1 and all(r == final_reward[0] for r in final_reward):
        final_reward = [r + random.uniform(-0.1, 0.1) for r in final_reward]

    # Log individual components manually (optional)
    try:
        wandb.log({
            "reward/final_mean": sum(final_reward)/len(final_reward),
            "reward/step_mean": sum(step_rewards)/len(step_rewards),
            "reward/answer_mean": sum(answer_rewards)/len(answer_rewards),
            "reward/format_mean": sum(format_rewards)/len(format_rewards)
        })
    except Exception:
        # Wandb not initialized, skip logging
        pass

    # Return final_reward list for GRPOTrainer to use
    return final_reward

