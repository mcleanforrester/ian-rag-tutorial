"""
Tool executor — routes a tool_use block from Claude to the correct Python function.

Claude's response contains tool_use blocks like:
  {"type": "tool_use", "id": "toolu_abc", "name": "calculator", "input": {"operation": "add", "a": 3, "b": 4}}

This module takes the name and input dict and returns a result string.
The result is what gets sent back to Claude as a tool_result message.
"""

from tools.definitions import calculator, employee_lookup


def execute_tool(tool_name: str, tool_input: dict) -> str:
    """
    Dispatch a tool call by name and return the result as a string.
    Claude will receive this string as the tool_result content.
    """
    if tool_name == "calculator":
        return calculator(
            operation=tool_input["operation"],
            a=tool_input["a"],
            b=tool_input["b"],
        )
    elif tool_name == "employee_lookup":
        return employee_lookup(employee_name=tool_input["employee_name"])
    else:
        return f"Error: unknown tool '{tool_name}'"
