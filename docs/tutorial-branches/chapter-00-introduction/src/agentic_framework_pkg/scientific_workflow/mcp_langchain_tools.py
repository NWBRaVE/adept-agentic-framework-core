from langchain.tools import BaseTool
from pydantic import Field, BaseModel
from typing import Type, Optional, Any, Dict, TypeVar
import asyncio
from fastmcp import Client as MCPClient
import os

class MCPToolWrapper(BaseTool):
    mcp_client_url: str
    actual_tool_name: str
    args_schema: Optional[Type[BaseModel]] = None

    def _run(self, **kwargs: Any) -> str:
        return asyncio.run(self._arun(**kwargs))

    async def _arun(self, **kwargs: Any) -> Any:
        async with MCPClient(self.mcp_client_url) as client:
            result = await client.call_tool(self.actual_tool_name, arguments=kwargs)
            return result.content # Return the raw content, not str(result)

class SQLQueryInput(BaseModel):
    query: str = Field(description="The SQL query to execute.")

def get_mcp_sql_tool_langchain():
    return MCPToolWrapper(
        name="execute_sql",
        mcp_client_url=os.getenv("MCP_SERVER_URL", "http://agentic_mcp_server_ch00:8080/mcp"),
        actual_tool_name="execute_sql",
        description="Executes a SQL query against the database.",
        args_schema=SQLQueryInput,
    )

class IngestDataInput(BaseModel):
    file_path: str = Field(description="The path to the CSV or Excel file to ingest.")
    table_name: str = Field(description="The name of the table to create in the database.")

def get_mcp_ingest_data_tool_langchain():
    return MCPToolWrapper(
        name="ingest_data",
        mcp_client_url=os.getenv("MCP_SERVER_URL", "http://agentic_mcp_server_ch00:8080/mcp"),
        actual_tool_name="ingest_data",
        description="Ingests data from a CSV or Excel file into a SQL table.",
        args_schema=IngestDataInput,
    )

def get_mcp_sql_schema_tool_langchain():
    return MCPToolWrapper(
        name="get_sql_schema",
        mcp_client_url=os.getenv("MCP_SERVER_URL", "http://agentic_mcp_server_ch00:8080/mcp"),
        actual_tool_name="get_sql_schema",
        description="Retrieves the SQL schema of the ingested tables from the database.",
    )

class RAGQueryInput(BaseModel):
    query: str = Field(description="The natural language query to perform RAG on CSV data.")

def get_mcp_rag_tool_langchain():
    return MCPToolWrapper(
        name="query_csv_rag",
        mcp_client_url=os.getenv("MCP_SERVER_URL", "http://agentic_mcp_server_ch00:8080/mcp"),
        actual_tool_name="query_csv_rag",
        description="Performs a RAG query on ingested CSV data using embeddings to find relevant information.",
        args_schema=RAGQueryInput,
    )

def get_mcp_list_files_tool_langchain():
    return MCPToolWrapper(
        name="list_files",
        mcp_client_url=os.getenv("MCP_SERVER_URL", "http://agentic_mcp_server_ch00:8080/mcp"),
        actual_tool_name="list_files",
        description="Lists all files that have been uploaded and ingested into the system.",
    )
