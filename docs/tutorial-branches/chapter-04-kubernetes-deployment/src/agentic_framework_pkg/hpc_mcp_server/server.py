import os
from fastmcp import FastMCP

# Initialize FastMCP instance for the HPC server
hpc_mcp = FastMCP(
    name=os.getenv("HPC_MCP_SERVER_NAME", "AgenticFramework_HPC_MCP"),
    instructions="This MCP server provides tools for High-Performance Computing tasks, including Nextflow pipeline execution."
    # Add other FastMCP configurations if needed
)