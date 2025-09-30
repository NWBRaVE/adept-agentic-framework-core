import uvicorn
import asyncio
import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
import json

# Load environment variables
if os.path.exists(".env"):
    load_dotenv()
elif os.path.exists("../../.env"):
    load_dotenv(dotenv_path=os.path.join(os.getcwd(), "../../.env"))

from .server import sandbox_mcp
from .tools import code_execution_tool
from ..logger_config import get_logger

logger = get_logger(__name__)

# Optional: Middleware for logging requests, similar to other servers
async def sandbox_log_request_middleware(request: Request, call_next):
    req_body_bytes = await request.body()
    logger.info(f"Sandbox Server Request: {request.method} {request.url}")
    if req_body_bytes:
        try:
            logger.info(f"Sandbox Request body (JSON): {json.loads(req_body_bytes.decode())}")
        except Exception:
            logger.info(f"Sandbox Request body (raw): {req_body_bytes[:200].decode(errors='ignore')}...")

    async def new_receive():
        return {"type": "http.request", "body": req_body_bytes, "more_body": False}
    new_request = Request(request.scope, receive=new_receive)
    response = await call_next(new_request)
    logger.info(f"Sandbox Server Response: {response.status_code}")
    return response

def setup_sandbox_mcp_server(mcp_instance=sandbox_mcp) -> FastAPI:
    logger.info("Setting up Sandbox MCP server and registering tools...")
    code_execution_tool.register_tools(mcp_instance)
    logger.info("Sandbox MCP tools registered.")

    actual_fastapi_app = mcp_instance.http_app()

    # Create a directory for public static files if it doesn't exist
    public_dir = "/app/public"
    os.makedirs(public_dir, exist_ok=True)

    # Mount the static files directory to be served at /static
    actual_fastapi_app.mount("/static", StaticFiles(directory=public_dir), name="static")
    logger.info(f"Serving static files from '{public_dir}' at the /static endpoint.")

    actual_fastapi_app.middleware("http")(sandbox_log_request_middleware)
    return actual_fastapi_app

async def sandbox_main_async():
    app = setup_sandbox_mcp_server()

    server_host = os.getenv("SANDBOX_MCP_SERVER_HOST", "0.0.0.0")
    server_port = int(os.getenv("SANDBOX_MCP_SERVER_PORT", "8082"))
    num_workers = int(os.getenv("SANDBOX_UVICORN_WORKERS", "1"))

    config = uvicorn.Config(app, host=server_host, port=server_port, log_level=os.getenv("LOG_LEVEL", "info").lower(), workers=num_workers)
    server = uvicorn.Server(config)
    logger.info(f"Starting Sandbox MCP Server on {server_host}:{server_port} with {num_workers} worker(s)...")
    await server.serve()

def start_sandbox_server_cli():
    try:
        asyncio.run(sandbox_main_async())
    except KeyboardInterrupt:
        logger.info("Sandbox MCP Server shutting down...")
    except Exception as e:
        logger.critical(f"Sandbox MCP Server failed to start or crashed: {e}", exc_info=True)

if __name__ == "__main__":
    start_sandbox_server_cli()


