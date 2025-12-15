"""Answer correctness reward functions."""

import re
from src.config import ANSWER_SEPARATOR, SCALING_FACTOR

match_numbers = re.compile(
    rf"{re.escape(ANSWER_SEPARATOR)}.*?([\d\.]{{1,}})",  # Extract numbers from solution section
    flags=re.MULTILINE | re.DOTALL        # Flexible pattern matching
)


def check_answer_correctness(completions, answer, **kwargs):
    """
    Robust numeric comparison reward:
      +15.0 exact match (within tolerance)
      -5.0 incorrect numeric answer
       0.0 if no answer extracted

    Tolerance: absolute error < 1e-6 or relative error < 1e-4
    """
    responses = [completion[0]["content"] for completion in completions]
    scores = []

    def safe_parse_number(x):
        """Try parsing integer/float; return None if impossible."""
        try:
            return float(x)
        except:
            return None

    for resp, true_answer in zip(responses, answer):
        # --- Extract model answer ---
        match = match_numbers.search(resp)
        if match is None:
            scores.append(0.0)
            continue

        extracted_raw = match.group(1).strip()
        gold_raw = true_answer.strip()

        # --- Parse both sides into floats safely ---
        pred = safe_parse_number(extracted_raw)
        gold = safe_parse_number(gold_raw)

        if pred is None or gold is None:
            # fallback: string exact match
            if extracted_raw == gold_raw:
                scores.append(3.0)
            else:
                scores.append(-0.5)
            continue

        # --- Compare with tolerance ---
        abs_err = abs(pred - gold)
        rel_err = abs_err / (abs(gold) + 1e-9)

        if abs_err < 1e-6 or rel_err < 1e-4:
            final = 15.0    # correct within tolerance
        else:
            final = -5.0   # wrong number

        scores.append(final / SCALING_FACTOR)

    return scores

