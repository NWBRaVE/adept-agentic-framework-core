from typing import Callable, Type, Optional, Any
from pydantic import BaseModel
from fastmcp import FastMCP, Context

def mcp_tool_factory(
    mcp_instance: FastMCP,
    name: str,
    description: str,
) -> Callable:
    """
    A factory function that returns a decorator for registering asynchronous functions
    as FastMCP tools.

    This factory simplifies tool creation by allowing the definition of a Pydantic
    schema for arguments, which is then automatically used by FastMCP.

    Args:
        mcp_instance: The FastMCP instance to register the tool with.
        name: The name of the tool (used for calling it).
        description: A description of what the tool does.
        args_schema: An optional Pydantic BaseModel class defining the tool's arguments.
                     If provided, FastMCP will automatically validate and parse arguments.

    Returns:
        A decorator that takes an asynchronous function (the tool's logic) and
        registers it with the FastMCP instance.
    """
    def decorator(func: Callable) -> Callable:
        """
        The actual decorator that wraps the user's asynchronous tool logic function.
        """
        # FastMCP's @mcp.tool() decorator handles argument parsing and validation
        # based on the decorated function's signature.
        return mcp_instance.tool(name=name, description=description)(func)
    return decorator
