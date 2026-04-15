"""
Tool definitions for the Anthropic API.

Each entry in TOOL_SCHEMAS is a dict Claude receives before the conversation.
It tells the model: what's the tool called, what does it do, and what arguments
does it take. Claude uses this to decide when to call a tool and what to pass.

The actual Python functions live here too — they're completely separate from the
schemas. Claude never sees the function body; it only sees the schema.
"""

import json

# ── Mock data ─────────────────────────────────────────────────────────────────

EMPLOYEES = [
    {"id": 1, "name": "Alice", "department": "engineering", "role": "senior engineer",    "manager": "Bob"},
    {"id": 2, "name": "Bob",   "department": "engineering", "role": "engineering lead",   "manager": "Carol"},
    {"id": 3, "name": "Carol", "department": "operations",  "role": "VP of operations",   "manager": None},
    {"id": 4, "name": "Dave",  "department": "sales",       "role": "account executive",  "manager": "Carol"},
]

# ── Python functions ───────────────────────────────────────────────────────────

def calculator(operation: str, a: float, b: float) -> str:
    if operation == "add":
        return str(a + b)
    elif operation == "subtract":
        return str(a - b)
    elif operation == "multiply":
        return str(a * b)
    elif operation == "divide":
        if b == 0:
            return "Error: division by zero"
        return str(a / b)
    else:
        return f"Error: unknown operation '{operation}'"


def employee_lookup(employee_name: str) -> str:
    match = next(
        (e for e in EMPLOYEES if e["name"].lower() == employee_name.lower()),
        None,
    )
    if match is None:
        return f"No employee found with name '{employee_name}'"
    return json.dumps(match)


# ── Schemas sent to the Anthropic API ─────────────────────────────────────────
#
# This is the format Anthropic expects. Each tool has:
#   name        — how Claude refers to it in a tool_use block
#   description — plain English; Claude reads this to decide when to use it
#   input_schema — JSON Schema describing the arguments
#
# Claude does NOT call the function. It produces a tool_use block with the
# name and arguments, then stops. Your code calls the function.

TOOL_SCHEMAS = [
    {
        "name": "calculator",
        "description": (
            "Performs basic arithmetic. Use this whenever the user asks you to "
            "add, subtract, multiply, or divide numbers."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                    "description": "The arithmetic operation to perform.",
                },
                "a": {"type": "number", "description": "The first operand."},
                "b": {"type": "number", "description": "The second operand."},
            },
            "required": ["operation", "a", "b"],
        },
    },
    {
        "name": "employee_lookup",
        "description": (
            "Looks up an employee record by name. Returns their department, role, "
            "and manager. Use this when the user asks about a specific employee."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "employee_name": {
                    "type": "string",
                    "description": "The full name of the employee to look up.",
                },
            },
            "required": ["employee_name"],
        },
    },
]
