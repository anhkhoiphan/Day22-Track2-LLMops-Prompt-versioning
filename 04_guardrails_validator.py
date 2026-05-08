"""
Step 4 - Guardrails AI Validators
====================================
Validator A - PIIDetector: detects and redacts emails, phone numbers,
  SSNs, and credit card numbers using regex.
Validator B - JSONFormatter: validates and auto-repairs malformed JSON
  (strips markdown fences, fixes single quotes, removes trailing commas).

Deliverable: all demo test cases pass with expected outputs logged.
"""

import re
import json
import sys

# Force UTF-8 output so logs save correctly on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# ── Imports with version-compatibility fallbacks ──────────────────────────────
from guardrails import Guard

try:
    from guardrails.validator_base import OnFailAction, Validator, register_validator
except ImportError:
    from guardrails.validators import OnFailAction, Validator, register_validator  # type: ignore

try:
    from guardrails.validators import PassResult, FailResult
except ImportError:
    from guardrails.validator_base import PassResult, FailResult  # type: ignore


# ── Validator A: PII Detector ─────────────────────────────────────────────────
@register_validator(name="custom/pii-detector", data_type="string")
class PIIDetector(Validator):
    """
    Detects and redacts Personally Identifiable Information (PII).
    Patterns: email, US phone, SSN, credit card.
    """

    PII_PATTERNS = {
        "EMAIL":       r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
        "PHONE":       r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b",
        "SSN":         r"\b\d{3}-\d{2}-\d{4}\b",
        "CREDIT_CARD": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    }

    def validate(self, value: str, metadata: dict):
        redacted_text = value
        found_pii     = []

        for pii_type, pattern in self.PII_PATTERNS.items():
            matches = re.findall(pattern, value)
            for match in matches:
                redacted_text = redacted_text.replace(match, f"[{pii_type}_REDACTED]")
                found_pii.append((pii_type, match))

        if found_pii:
            print(f"  [WARN] Redacted {len(found_pii)} PII item(s): {[p[0] for p in found_pii]}")
            return PassResult(value_override=redacted_text)
        return PassResult(value_override=value)


# ── Validator B: JSON Formatter ───────────────────────────────────────────────
@register_validator(name="custom/json-formatter", data_type="string")
class JSONFormatter(Validator):
    """
    Validates and auto-repairs malformed JSON strings.
    Repairs: strip markdown fences, single->double quotes, trailing commas.
    """

    @staticmethod
    def _repair(text: str) -> str:
        text = text.strip()
        # Remove markdown code fences
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$",          "", text)
        text = text.strip()
        # Single quotes -> double quotes
        text = text.replace("'", '"')
        # Remove trailing commas before } or ]
        text = re.sub(r",\s*([}\]])", r"\1", text)
        return text

    def validate(self, value: str, metadata: dict):
        # Try parsing as-is
        try:
            parsed = json.loads(value)
            return PassResult(value_override=json.dumps(parsed, indent=2))
        except json.JSONDecodeError:
            pass

        # Try repair
        try:
            repaired_text = self._repair(value)
            parsed        = json.loads(repaired_text)
            print("  [FIX] JSON repaired successfully")
            return PassResult(value_override=json.dumps(parsed, indent=2))
        except json.JSONDecodeError as e:
            return FailResult(
                error_message=f"Invalid JSON after repair attempt: {e}",
                fix_value=json.dumps({"error": str(e), "raw": value[:200]}),
            )


# ── PII Guard demo ────────────────────────────────────────────────────────────
def demo_pii_guard():
    print("\n" + "=" * 55)
    print("  PII Detection Demo")
    print("=" * 55)

    guard = Guard().use(PIIDetector(on_fail=OnFailAction.FIX))

    test_cases = [
        ("Email",       "Contact John at john.doe@example.com for details."),
        ("Phone",       "Call our support line at (555) 867-5309."),
        ("SSN",         "Patient SSN is 123-45-6789 on file."),
        ("Credit Card", "Payment made with card 4532 1234 5678 9010."),
        ("Multi-PII",   "Email: alice@example.com, Phone: 555-123-4567"),
        ("Clean",       "No sensitive information in this text."),
    ]

    for label, text in test_cases:
        result = guard.validate(text)
        print(f"\n[{label}]")
        print(f"  Input:  {text}")
        print(f"  Output: {result.validated_output}")


# ── JSON Guard demo ───────────────────────────────────────────────────────────
def demo_json_guard():
    print("\n" + "=" * 55)
    print("  JSON Formatting Demo")
    print("=" * 55)

    guard = Guard().use(JSONFormatter(on_fail=OnFailAction.FIX))

    test_cases = [
        ("Valid JSON",      '{"name": "Alice", "age": 30}'),
        ("Markdown fences", '```json\n{"name": "Bob"}\n```'),
        ("Single quotes",   "{'name': 'Charlie', 'score': 95}"),
        ("Trailing comma",  '{"key": "value",}'),
        ("Truly invalid",   "This is not JSON at all: ??? {]"),
    ]

    for label, text in test_cases:
        result = guard.validate(text)
        status = "[PASS]" if result.validation_passed else "[FAIL]"
        print(f"\n[{label}] {status}")
        print(f"  Input:  {text[:60]}")
        print(f"  Output: {str(result.validated_output)[:80]}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  Step 4: Guardrails AI Validators")
    print("=" * 55)

    demo_pii_guard()
    demo_json_guard()

    print("\n[DONE] Step 4 complete!")


if __name__ == "__main__":
    main()
