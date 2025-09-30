import argparse
import asyncio
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from agentic_framework_pkg.agent_gateway.keycloak_admin_client import KeycloakAdminClient

async def main():
    """Creates a new service client in Keycloak."""
    parser = argparse.ArgumentParser(description="Create a new service client in Keycloak.")
    parser.add_argument("--client-name", required=True, help="The name for the new service client.")
    args = parser.parse_args()

    keycloak_admin = KeycloakAdminClient()
    try:
        client_info = keycloak_admin.create_service_client(args.client_name)
        print(f"Service client '{args.client_name}' created successfully:")
        print(f"  Client ID: {client_info['client_id']}")
        print(f"  Client Secret: {client_info['client_secret']}")
    except Exception as e:
        print(f"Error creating service client: {e}")

if __name__ == "__main__":
    asyncio.run(main())
