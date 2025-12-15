"""Stepwise reward functions for mathematical reasoning."""

import re
import random
from sympy import solve, Eq
from sympy.logic.boolalg import BooleanTrue, BooleanFalse

from src.config import REASONING_START, REASONING_END, EQUATION_SEPARATOR, SCALING_FACTOR
from src.rewards.equation_parser import parse_numeric_or_symbolic_equation

# Regex Patterns
STEP_PATTERN = re.compile(r"<STEP(?:\s*\d+)?>\s*(.*?)\s*</STEP>", re.DOTALL)
EQUATION_PATTERN = re.compile(rf"\@\@\@\@(.*?)\@\@\@\@", re.DOTALL)


def extract_steps(text):
    """Extract steps from reasoning text."""
    return STEP_PATTERN.findall(text)


def extract_equations(step):
    """Extract equations from a step."""
    return [eq.strip() for eq in EQUATION_PATTERN.findall(step)]


def symbolic_follows(prev_eq, curr_eq, tol=1e-6):
    """
    Check if curr_eq logically follows prev_eq using solve.
    Handles single-variable equations efficiently.
    """
    syms = list(prev_eq.free_symbols.union(curr_eq.free_symbols))

    if not syms:
        # No symbols: check numeric difference
        return abs(float(prev_eq.lhs - prev_eq.rhs) - float(curr_eq.lhs - curr_eq.rhs)) <= tol

    if len(syms) == 1:
        var = syms[0]
        try:
            solutions = solve(prev_eq, var)
        except Exception:
            return False
        if not solutions:
            return False
        for sol in solutions:
            val = curr_eq.subs(var, sol)
            if isinstance(val, (BooleanTrue, BooleanFalse)):
                if val != True:  # False is not satisfied
                    return False
            else:
                if abs(float(val)) > tol:
                    return False

        return True
    else:
        # Multi-variable fallback
        num_tests = 4
        for _ in range(num_tests):
            vals = {s: random.uniform(1, 10) for s in syms}
            prev_val = float((prev_eq.lhs - prev_eq.rhs).subs(vals))
            curr_val = float((curr_eq.lhs - curr_eq.rhs).subs(vals))
            if abs(prev_val - curr_val) > tol:
                return False
        return True


def numeric_equation_correct(lhs, rhs, tol=1e-9):
    """Check if numeric equation is correct."""
    return abs(lhs - rhs) <= tol


def numeric_partial_credit(lhs, rhs, rel_tol=0.05):
    """Check if numeric equation gets partial credit."""
    if rhs == 0:
        return abs(lhs - rhs) <= rel_tol
    return abs(lhs - rhs) / abs(rhs) <= rel_tol


def numeric_step_complexity(lhs_str, rhs_str):
    """Count the number of arithmetic operations in the equation."""
    if not lhs_str or not rhs_str:
        return None
    equation_text = lhs_str + "=" + rhs_str
    num_ops = len(re.findall(r"[\+\-\*/\^]", equation_text))
    return num_ops


def is_exact_repeat(lhs_rhs_strs, previous_equations):
    """Check if equation is an exact repeat (including flipped)."""
    lhs_str, rhs_str = lhs_rhs_strs
    if lhs_str is None or rhs_str is None:
        return False  # Malformed equations never count
    for prev_lhs, prev_rhs in previous_equations:
        if (lhs_str == prev_lhs and rhs_str == prev_rhs) or (lhs_str == prev_rhs and rhs_str == prev_lhs):
            return True
    return False


def reward_stepwise(completions, answer, gamma=0.9, **kwargs):
    """Compute stepwise rewards for mathematical reasoning."""
    rewards = []

    for comp_idx, (comp, gold_ans) in enumerate(zip(completions, answer)):
        text = comp[0]["content"]
        raw_steps = extract_steps(text)

        if not raw_steps:
            rewards.append(0.0)
            continue

        # Only keep steps with equations
        step_data = []
        for idx, step in enumerate(raw_steps):
            equations = extract_equations(step)
            if not equations:
                continue
            eq_str = equations[-1]
            lhs_val, rhs_val, lhs_str, rhs_str, malformed, symbolic_eq = parse_numeric_or_symbolic_equation(eq_str)
            complexity = numeric_step_complexity(lhs_str, rhs_str)
            step_data.append((idx+1, (lhs_str, rhs_str), (lhs_val, rhs_val), malformed, symbolic_eq, complexity))

        total_reward = 0.0

        # Separate histories for numeric and symbolic
        prev_numeric_equations = []
        last_numeric_complexity = None

        prev_symbolic_equations = []
        last_symbolic_complexity = None
        last_valid_symbolic_eq = None

        for idx, (step_no, lhs_rhs_strs, lhs_rhs_vals, malformed, symbolic_eq, complexity) in enumerate(step_data):
            step_reward = 0.0

            if malformed:
                step_reward -= 1.0
            else:
                lhs_val, rhs_val = lhs_rhs_vals
                # -----------------------
                # Numeric equations
                # -----------------------
                if lhs_val is not None and rhs_val is not None:
                    if numeric_equation_correct(lhs_val, rhs_val):
                        step_reward += 2.0
                    elif numeric_partial_credit(lhs_val, rhs_val):
                        step_reward += 0.5
                    else:
                        step_reward -= 1.0

                    # Complexity check
                    if last_numeric_complexity is not None and complexity is not None:
                        if complexity < last_numeric_complexity:
                            step_reward += 0.5
                        elif complexity > last_numeric_complexity:
                            step_reward -= 0.5

                    # Repetition check
                    if is_exact_repeat(lhs_rhs_strs, prev_numeric_equations):
                        step_reward -= 1.0

                    # Update numeric history
                    if complexity is not None:
                        prev_numeric_equations.append(lhs_rhs_strs)
                        last_numeric_complexity = complexity

                # -----------------------
                # Symbolic equations
                # -----------------------
                elif symbolic_eq is not None:
                    if last_valid_symbolic_eq is None:
                        step_reward += 1.0
                    else:
                        if symbolic_follows(last_valid_symbolic_eq, symbolic_eq):
                            step_reward += 2.0
                        else:
                            step_reward -= 0.5

                    # Complexity check
                    if last_symbolic_complexity is not None and complexity is not None:
                        if complexity < last_symbolic_complexity:
                            step_reward += 0.5
                        elif complexity > last_symbolic_complexity:
                            step_reward -= 0.5

                    # Repetition check
                    if is_exact_repeat(lhs_rhs_strs, prev_symbolic_equations):
                        step_reward -= 1.0

                    # Update symbolic history
                    last_valid_symbolic_eq = symbolic_eq
                    if complexity is not None:
                        prev_symbolic_equations.append(lhs_rhs_strs)
                        last_symbolic_complexity = complexity

            # Apply diminishing returns
            if step_reward > 0:
                step_reward *= gamma ** idx

            total_reward += step_reward

        rewards.append(total_reward / SCALING_FACTOR)

    return rewards

