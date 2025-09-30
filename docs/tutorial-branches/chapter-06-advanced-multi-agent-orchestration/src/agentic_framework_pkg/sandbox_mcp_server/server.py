import os
from fastmcp import FastMCP

# Initialize FastMCP instance for the Sandbox server
sandbox_mcp = FastMCP(
    name=os.getenv("SANDBOX_MCP_SERVER_NAME", "AgenticFramework_Sandbox_MCP"),
    instructions="This MCP server provides a tool for executing Python code in a secure sandbox."
)


