# Guardrails Checklist

## Architecture/Big Picture
- Prompt Injection
- Role consistency
- Refusal rate monitoring
- Confidence thresholds
- Gating downstream actions for side effects like API calls - require an extra validation step?
- Consistency checks for high risk queries/outputs
- Rate limiting

## Immediate Validators
- Input sanitization (special characters that could malform the input)
- PII
- Length (to cut down on token usage)
- Source Citations?
- Content category filtering
- Relevance detection
- Etiquette validation?