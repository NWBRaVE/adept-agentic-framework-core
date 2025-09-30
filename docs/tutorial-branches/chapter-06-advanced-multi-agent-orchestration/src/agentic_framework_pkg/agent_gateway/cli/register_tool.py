import argparse
import asyncio
import json
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from agentic_framework_pkg.agent_gateway.keycloak_admin_client import KeycloakAdminClient
from agentic_framework_pkg.agent_gateway.app import tool_manager

async def main():
    """Registers a new tool with the Agent Gateway."""
    parser = argparse.ArgumentParser(description="Register a new tool with the Agent Gateway.")
    parser.add_argument("--tool-config", required=True, help="Path to the JSON file containing the tool configuration.")
    args = parser.parse_args()

    with open(args.tool_config, 'r') as f:
        tool_config = json.load(f)

    keycloak_admin = KeycloakAdminClient()
    try:
        # For simplicity, this script uses the admin credentials to get a token.
        # In a real-world scenario, you might have a dedicated service account for this.
        admin_jwt = await keycloak_admin.keycloak_admin.get_token()

        await tool_manager.register_tool(tool_config)

        print(f"Tool '{tool_config['name']}' registered successfully.")
    except Exception as e:
        print(f"Error registering tool: {e}")

if __name__ == "__main__":
    asyncio.run(main())
