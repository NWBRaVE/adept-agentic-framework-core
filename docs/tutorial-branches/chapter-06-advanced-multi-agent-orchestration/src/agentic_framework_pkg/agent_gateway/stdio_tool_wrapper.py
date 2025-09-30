import asyncio
import json
import os
from langchain.tools import Tool
from pydantic import BaseModel, Field
from typing import Type, Dict, Any, List, Optional

"""
This module provides the core functionality for creating and executing tools that
communicate over standard I/O (stdio). It defines a generic wrapper that can turn
any command-line executable that reads JSON from stdin and writes JSON to stdout
into a LangChain-compatible `Tool`.

### Architectural Role:

This wrapper is the key component that enables the "Lightweight Local Execution"
pattern described in the agent gateway's roadmap. It decouples the Agent Gateway
from the specific implementation of a tool, allowing it to execute tools written
in any language, as long as they adhere to the simple stdio JSON contract.

It is used by the `ExternalToolManager`'s `_create_stdio_tool_from_config` factory
to create tool instances for any registered tool that specifies `"protocol": "stdio"`.

### How It Works:

1.  **`create_stdio_tool` (Factory Function):**
    -   This function receives the tool's configuration (name, description, command,
      and Pydantic schema for arguments).
    -   It dynamically creates a coroutine (`tool_func`) that will be the core
      execution logic for the LangChain `Tool`.
    -   It returns a `langchain.tools.Tool` instance, passing the `tool_func` as
      its `coroutine`.

2.  **`_run_stdio_tool` (Execution Logic):**
    -   This async function is called by the `tool_func` when the agent decides to
      use the tool.
    -   It takes the command to execute (e.g., `["docker", "run", "my-image"]`)
      and the arguments provided by the LLM.
    -   It uses Python's `asyncio.create_subprocess_exec` to run the command as a
      separate process.
    -   It serializes the input arguments into a JSON string and writes them to the
      subprocess's `stdin`.
    -   It captures the `stdout` and `stderr` from the subprocess.
    -   If the process exits with an error, it returns a formatted error message.
    -   If successful, it returns the JSON string received from the tool's `stdout`.

This mechanism allows the Agent Gateway to seamlessly invoke a wide variety of
external processes, from simple Python scripts to complex Docker containers, as
if they were native Python tools.
"""

async def _run_stdio_tool(command: List[str], input_data: Dict[str, Any], working_directory: Optional[str] = None) -> str:
    """
    Asynchronously runs a command-line tool, passing JSON data via stdin
    and capturing JSON data from stdout.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=working_directory
        )

        input_json = json.dumps(input_data)
        stdout, stderr = await proc.communicate(input=input_json.encode())

        if proc.returncode != 0:
            error_message = stderr.decode().strip()
            return f"Error executing command '{command}': {error_message}"

        result_json = stdout.decode().strip()
        return result_json

    except FileNotFoundError:
        return f"Error: The command '{command[0]}' was not found."
    except Exception as e:
        return f"Failed to run stdio tool with command '{command}': {e}"

def create_stdio_tool(
    name: str,
    description: str,
    command: List[str],
    args_schema: Type[BaseModel],
    working_directory: Optional[str] = None
) -> Tool:
    """
    Factory function to create a LangChain Tool that executes a local command via stdio.
    """
    async def tool_func(tool_input: Any) -> str:
        # LangChain might pass arguments as a string or a dictionary
        if isinstance(tool_input, str):
            try:
                # Attempt to parse the string as JSON
                parsed_input = json.loads(tool_input)
            except json.JSONDecodeError:
                # If it's not JSON, assume it's the value for the first argument in args_schema
                # This is a heuristic and might need refinement for complex schemas
                if hasattr(args_schema, 'model_fields') and len(args_schema.model_fields) == 1:
                    first_field_name = list(args_schema.model_fields.keys())[0]
                    parsed_input = {first_field_name: tool_input}
                else:
                    # Fallback if schema is complex or not a single field
                    raise ValueError(f"Tool input is a string and not valid JSON, and args_schema is not a single field: {tool_input}")
        elif isinstance(tool_input, dict):
            parsed_input = tool_input
        else:
            raise TypeError(f"Unexpected tool input type: {type(tool_input)}. Expected str or dict.")

        return await _run_stdio_tool(command, parsed_input, working_directory)

    return Tool(
        name=name,
        description=description,
        func=None,
        coroutine=tool_func,
        args_schema=args_schema
    )

# Example of a Pydantic model for the echo tool
class EchoToolSchema(BaseModel):
    message: str = Field(..., description="The message to be echoed back.")
