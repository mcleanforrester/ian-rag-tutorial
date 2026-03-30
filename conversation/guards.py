from guardrails.hub import ValidLength
from guardrails import Guard


def _custom_failed_response(value, fail_result):
    return None


input_guard = Guard().use(
    ValidLength(min=10, max=200, on_fail=_custom_failed_response),
)


def validate_input(query: str) -> bool:
    """Return True if the input passes validation."""
    return input_guard.parse(query).validation_passed is not False
