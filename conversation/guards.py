from guardrails.hub import ValidLength
from guardrails import Guard
from pydantic import BaseModel, Field


class PolicySummary(BaseModel):
    title: str = Field(..., description="The name of the policy")
    department: str = Field(..., description="The department this policy belongs to")
    effective_date: str = Field(..., description="When this policy takes effect (or 'Unknown')")
    key_points: list[str] = Field(..., description="3-5 bullet points summarizing the policy")


def _custom_failed_response(value, fail_result):
    return None


input_guard = Guard().use(
    ValidLength(min=10, max=200, on_fail=_custom_failed_response),
)

output_guard = Guard.for_pydantic(output_class=PolicySummary)


def validate_input(query: str) -> bool:
    """Return True if the input passes validation."""
    return input_guard.parse(query).validation_passed is not False


def extract_structured(response: str) -> tuple[bool, object]:
    """Attempt structured extraction. Returns (success, validated_output_or_error)."""
    result = output_guard.parse(response)
    if result.validation_passed:
        return True, result.validated_output
    return False, result.error
