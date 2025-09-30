import os
from fastmcp import FastMCP

mcp = FastMCP(
    name=os.getenv("MCP_SERVER_NAME", "AgenticFrameworkMCP"),
    instructions="This MCP server provides tools for data analysis, general utilities, and scientific workflows."
)
