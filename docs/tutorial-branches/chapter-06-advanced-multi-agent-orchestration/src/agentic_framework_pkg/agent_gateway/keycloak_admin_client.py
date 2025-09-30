"""
This module provides a client for programmatically managing a Keycloak instance.
It is used by the command-line admin tools to create users, groups, and clients.
"""
import os
from keycloak import KeycloakAdmin
from keycloak.exceptions import KeycloakError
from typing import Dict, Any, List

from ..logger_config import get_logger

logger = get_logger(__name__)

class KeycloakAdminClient:
    """A client for interacting with the Keycloak Admin API."""

    def __init__(self):
        self.keycloak_admin = KeycloakAdmin(
            server_url=os.getenv("KEYCLOAK_URL", "http://localhost:8180/"),
            username=os.getenv("KEYCLOAK_ADMIN", "admin"),
            password=os.getenv("KEYCLOAK_ADMIN_PASSWORD", "admin"),
            realm_name=os.getenv("KEYCLOAK_REALM", "master"),
            user_realm_name="master",
            verify=True,
        )

    def create_user(self, username: str, password: str) -> str:
        """Creates a new user in Keycloak."""
        try:
            user_id = self.keycloak_admin.create_user(
                {"email": f"{username}@example.com", "username": username, "enabled": True}
            )
            self.keycloak_admin.set_user_password(user_id, password, temporary=False)
            logger.info(f"Successfully created user '{username}' with ID: {user_id}")
            return user_id
        except KeycloakError as e:
            logger.error(f"Failed to create user '{username}': {e}")
            raise

    def create_group(self, group_name: str) -> str:
        """Creates a new group in Keycloak."""
        try:
            group_id = self.keycloak_admin.create_group({"name": group_name})
            logger.info(f"Successfully created group '{group_name}' with ID: {group_id}")
            return group_id
        except KeycloakError as e:
            logger.error(f"Failed to create group '{group_name}': {e}")
            raise

    def add_user_to_group(self, user_id: str, group_id: str):
        """Adds a user to a group in Keycloak."""
        try:
            self.keycloak_admin.group_user_add(user_id, group_id)
            logger.info(f"Successfully added user '{user_id}' to group '{group_id}'")
        except KeycloakError as e:
            logger.error(f"Failed to add user '{user_id}' to group '{group_id}': {e}")
            raise

    def create_service_client(self, client_name: str) -> Dict[str, Any]:
        """Creates a new service account client in Keycloak."""
        try:
            client_id = self.keycloak_admin.create_client(
                {
                    "clientId": client_name,
                    "serviceAccountsEnabled": True,
                    "publicClient": False,
                    "directAccessGrantsEnabled": True,
                }
            )
            client_secret = self.keycloak_admin.get_client_secrets(client_id)[0]
            logger.info(f"Successfully created service client '{client_name}' with ID: {client_id}")
            return {"client_id": client_id, "client_secret": client_secret["value"]}
        except KeycloakError as e:
            logger.error(f"Failed to create service client '{client_name}': {e}")
            raise
