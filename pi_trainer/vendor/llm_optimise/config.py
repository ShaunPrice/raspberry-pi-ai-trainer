"""Extracted LLM-Optimise numeric validators. See LICENSE."""
import math

def finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{label} must be a finite number')

def positive(value, label, integer=False):
    finite(value, label)
    if value <= 0 or (integer and type(value) is not int):
        raise ValueError(f"{label} must be a positive {'integer' if integer else 'number'}")
