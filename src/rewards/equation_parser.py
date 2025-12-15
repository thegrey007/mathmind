"""Equation parsing utilities for stepwise rewards."""

import re
from sympy import symbols, Eq, sympify


def parse_numeric_or_symbolic_equation(eq_str):
    """
    Parse a numeric or symbolic equation string and return:
    - lhs_val, rhs_val (float) if numeric evaluation succeeds, else None
    - lhs_str, rhs_str (normalized strings)
    - malformed (bool)
    - symbolic_eq (sympy Eq object if numeric fails or contains symbols, else None)
    """
    malformed = False
    symbolic_eq = None

    # ------------------------------------------------------------
    # 0. Basic malformed checks
    # ------------------------------------------------------------
    num_equals = eq_str.count("=")
    if num_equals != 1:
        return None, None, None, None, True, None

    lhs_str, rhs_str = eq_str.split("=", 1)
    lhs_str = lhs_str.strip()
    rhs_str = rhs_str.strip()
    if not lhs_str or not rhs_str:
        return None, None, None, None, True, None

    # ------------------------------------------------------------
    # 1. Normalize unicode operators
    # ------------------------------------------------------------
    def normalize_unicode(s):
        return s.replace("×", "*").replace("÷", "/").replace("−", "-")

    lhs_str = normalize_unicode(lhs_str)
    rhs_str = normalize_unicode(rhs_str)

    # ------------------------------------------------------------
    # 2. Remove LaTeX-style operators
    # ------------------------------------------------------------
    def remove_simple_latex(s):
        s = s.replace("\\times", "*")
        s = s.replace("\\cdot", "*")
        s = s.replace("\\div", "/")
        s = s.replace("\\left", "").replace("\\right", "")
        return s

    lhs_str = remove_simple_latex(lhs_str)
    rhs_str = remove_simple_latex(rhs_str)

    # ------------------------------------------------------------
    # 3. Convert LaTeX fractions \frac{a}{b} -> (a)/(b)
    # ------------------------------------------------------------
    def convert_frac(s):
        pattern = r"\\frac\s*{\s*(.*?)\s*}\s*{\s*(.*?)\s*}"
        while re.search(pattern, s):
            s = re.sub(pattern, r"(\1)/(\2)", s)
        return s

    lhs_str = convert_frac(lhs_str)
    rhs_str = convert_frac(rhs_str)

    # ------------------------------------------------------------
    # 4. Cleanup braces and whitespace
    # ------------------------------------------------------------
    def cleanup(s):
        s = s.replace("{", "(").replace("}", ")").replace("\\", "")
        s = " ".join(s.split())
        return s

    lhs_str = cleanup(lhs_str)
    rhs_str = cleanup(rhs_str)

    # ------------------------------------------------------------
    # 5. Insert implicit multiplication
    # ------------------------------------------------------------
    def insert_implicit_mul(s):
        # number or closing parenthesis followed by a symbol: 2x -> 2*x, (1/3)x -> (1/3)*x
        s = re.sub(r'(\d|\))([a-zA-Z])', r'\1*\2', s)
        # symbol followed by opening parenthesis: x(x+1) -> x*(x+1)
        s = re.sub(r'([a-zA-Z])\(', r'\1*(', s)
        return s

    lhs_str = insert_implicit_mul(lhs_str)
    rhs_str = insert_implicit_mul(rhs_str)

    # ------------------------------------------------------------
    # 6. Parse symbolically first
    # ------------------------------------------------------------
    lhs_val = rhs_val = None
    try:
        lhs_sym = sympify(lhs_str)
        rhs_sym = sympify(rhs_str)
        symbolic_eq = Eq(lhs_sym, rhs_sym)

        # If there are no free symbols, try numeric evaluation
        if not symbolic_eq.free_symbols:
            lhs_val = float(lhs_sym)
            rhs_val = float(rhs_sym)
            symbolic_eq = None

    except Exception:
        malformed = True
        lhs_val = rhs_val = None
        symbolic_eq = None

    return lhs_val, rhs_val, lhs_str, rhs_str, malformed, symbolic_eq

