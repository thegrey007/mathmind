"""Format checking reward functions."""

import re
from src.config import REASONING_START, REASONING_END, ANSWER_SEPARATOR, SCALING_FACTOR

# Compiled regex patterns for efficient reward computation
match_format = re.compile(
    rf"^[\s]{{0,}}"                      # Optional whitespace at start
    rf"{re.escape(REASONING_START)}.+?{re.escape(REASONING_END)}.*?"  # Reasoning section (non-greedy)
    rf"[\s]{{0,}}$",                     # Optional whitespace at end
    flags=re.MULTILINE | re.DOTALL       # Multi-line matching with . matching newlines
)

match_numbers = re.compile(
    rf"{re.escape(ANSWER_SEPARATOR)}.*?([\d\.]{{1,}})",  # Extract numbers from solution section
    flags=re.MULTILINE | re.DOTALL        # Flexible pattern matching
)


def check_format_rewards(completions, answer, **kwargs):
    """
    Rewards:
      +1 reasoning block exists using match_format
      +1 steps present AND strictly increasing (1..N)
      +1 final answer is numeric and extracted by match_numbers
    """
    scores = []

    # Step regex (inside reasoning section)
    step_pattern = re.compile(r"<STEP\s+(\d+)>.*?</STEP>", re.DOTALL)

    for completion in completions:
        response = completion[0]["content"]
        score = 0

        # -----------------------------
        # 1. Check reasoning block
        # -----------------------------
        reasoning_match = match_format.search(response)
        if reasoning_match:
            score += 1  # reasoning block exists

            reasoning_text = reasoning_match.group(0)

            # -----------------------------
            # 2. Check steps (strictly 1,2,3,...)
            # -----------------------------
            steps = step_pattern.findall(reasoning_text)
            if steps:
                steps = list(map(int, steps))
                if steps == list(range(1, len(steps) + 1)):
                    score += 1

        # -----------------------------
        # 3. Check final numeric answer
        # -----------------------------
        number_match = match_numbers.search(response)
        if number_match:
            score += 1

        if score == 0:
            final = -10.0
        else:
            final = float(score)

        scores.append(final / SCALING_FACTOR)

    return scores

