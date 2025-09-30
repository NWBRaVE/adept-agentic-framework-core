import uvicorn
import asyncio
import os

from .server import mcp
from .state_manager import initialize_db
from .tools import knowledge_base_tool
from .tools import text_analysis_tool # Import the new tool
from ..core.llm_agnostic_layer import LLMAgnosticClient
from ..core.logger_config import get_logger

logger = get_logger(__name__)


llm_agnostic_client_instance = LLMAgnosticClient()

def setup_mcp_server():
    logger.info("Setting up MCP server and registering tools...")
    knowledge_base_tool.register_tools(mcp) # This now registers ingest_data
    text_analysis_tool.register_text_analysis_tool(mcp) # Register the new tool
    logger.info("MCP tools registered.")
    return mcp.http_app()

async def main_async():
    logger.info("Initializing database...")
    await initialize_db()
    
    app = setup_mcp_server()

    server_host = os.getenv("MCP_SERVER_HOST", "0.0.0.0")
    server_port = int(os.getenv("MCP_SERVER_PORT", "8080"))

    config = uvicorn.Config(
        app,
        host=server_host,
        port=server_port,
        log_level=os.getenv("LOG_LEVEL", "INFO").lower(),
    )
    server = uvicorn.Server(config)
    logger.info(f"Starting MCP Server on {server_host}:{server_port}...")
    await server.serve()

def start_server_cli():
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        logger.info("MCP Server shutting down...")
    except Exception as e:
        logger.critical(f"MCP Server failed to start or crashed: {e}", exc_info=True)

if __name__ == "__main__":
    start_server_cli()
