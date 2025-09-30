import uvicorn
import os
from ..logger_config import get_logger

logger = get_logger(__name__)

def start_agent_gateway():
    """CLI entry point to start the Agent Gateway server."""
    port = int(os.getenv("AGENT_GATEWAY_PORT", "8081"))
    host = os.getenv("AGENT_GATEWAY_HOST", "0.0.0.0")
    
    logger.info(f"Starting Agent Gateway server on {host}:{port}")
    
    uvicorn.run(
        "agentic_framework_pkg.agent_gateway.app:app",
        host=host,
        port=port,
        reload=True, # Reload server on code changes, useful for development
    )

if __name__ == "__main__":
    start_agent_gateway()