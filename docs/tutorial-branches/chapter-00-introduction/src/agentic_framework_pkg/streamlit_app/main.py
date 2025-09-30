import subprocess
import os
import sys

from ..core.logger_config import get_logger
logger = get_logger(__name__)

def start_harness_cli():
    # Dynamically determine the path to the .env file
    # This is no longer strictly necessary if using docker-compose env_file, but good for local dev outside docker
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    # project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
    # dotenv_path = os.path.join(project_root, ".env")

    # if os.path.exists(dotenv_path):
    #     logger.info(f"Loading .env file from: {dotenv_path}")
    #     load_dotenv(dotenv_path=dotenv_path)
    # else:
    #     logger.warning(f".env file not found at: {dotenv_path}. Environment variables might not be loaded.")

    app_py_path = os.path.join(os.path.dirname(__file__), "app.py")
    
    port = os.getenv("STREAMLIT_SERVER_PORT", "8501")
    address = os.getenv("STREAMLIT_SERVER_ADDRESS", "0.0.0.0")

    cmd = [
        "streamlit", "run", app_py_path,
        "--server.port", port,
        "--server.address", address,
        "--server.headless", "true",
        "--server.enableCORS", "false",
    ]

    if os.getenv("STREAMLIT_ENABLE_SSL", "false").lower() == "true":
        cmd.extend([
            "--server.sslCertFile", "/app/certs/cert.pem",
            "--server.sslKeyFile", "/app/certs/key.pem",
        ])
    
    logger.info(f"Starting Streamlit harness with command: {' '.join(cmd)}")
    try:
        process = subprocess.Popen(cmd)
        process.wait()
    except FileNotFoundError:
        err_msg = "'streamlit' command not found. Make sure Streamlit is installed and in your PATH."
        logger.critical(err_msg)
        print(f"Error: {err_msg}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Error running Streamlit: {e}", exc_info=True)
        print(f"Error running Streamlit: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    start_harness_cli()
