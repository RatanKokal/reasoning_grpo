"""Answer extraction and numeric equivalence for GSM8K-style tasks."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction

_ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
_FINAL_RE = re.compile(r"final\s+answer\s*[:=]\s*(.+)", re.IGNORECASE)
_BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
_HASH_RE = re.compile(r"####\s*(.+)")
_NUMBER_RE = re.compile(r"[-+]?\$?\d[\d,]*(?:\.\d+)?(?:/[-+]?\d[\d,]*)?")


def extract_gsm8k_reference(answer: str) -> tuple[str, str]:
    """Return (reasoning, final answer) from the canonical GSM8K answer field."""
    if "####" not in answer:
        return answer.strip(), extract_final_answer(answer) or ""
    reasoning, final = answer.rsplit("####", 1)
    return reasoning.strip(), normalize_answer(final)


def extract_final_answer(text: str) -> str | None:
    """Extract a final answer using explicit formats first, then the last number."""
    if not text:
        return None

    tagged = _ANSWER_TAG_RE.findall(text)
    if tagged:
        return normalize_answer(tagged[-1])

    boxed = _BOXED_RE.findall(text)
    if boxed:
        return normalize_answer(boxed[-1])

    hashed = _HASH_RE.findall(text)
    if hashed:
        return normalize_answer(hashed[-1])

    final_lines = _FINAL_RE.findall(text)
    if final_lines:
        return normalize_answer(final_lines[-1])

    numbers = _NUMBER_RE.findall(text)
    if numbers:
        return normalize_answer(numbers[-1])
    return None


def normalize_answer(value: object) -> str:
    """Normalize common numeric formatting without evaluating arbitrary expressions."""
    text = str(value).strip()
    text = text.replace("$", "").replace(",", "")
    text = text.rstrip(". ")
    text = re.sub(r"\s+", "", text)
    return text


def _as_fraction(value: str) -> Fraction | None:
    value = normalize_answer(value)
    try:
        if "/" in value:
            numerator, denominator = value.split("/", 1)
            return Fraction(Decimal(numerator)) / Fraction(Decimal(denominator))
        return Fraction(Decimal(value))
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None


def answers_equal(prediction: str | None, reference: str | None) -> bool:
    """Compare normalized text, then exact rational numeric values when possible."""
    if prediction is None or reference is None:
        return False
    pred = normalize_answer(prediction)
    ref = normalize_answer(reference)
    if pred.casefold() == ref.casefold():
        return True
    pred_number = _as_fraction(pred)
    ref_number = _as_fraction(ref)
    return pred_number is not None and ref_number is not None and pred_number == ref_number

