"""
This module handles JWT-based authentication and authorization for the Agent Gateway.

It provides utilities to:
1.  Fetch public keys from a Keycloak OIDC endpoint.
2.  Decode and validate incoming JWTs.
3.  Extract user identity and permissions from a valid token.
"""
import os
import httpx
from jose import jwt, jwk
from jose.exceptions import JOSEError
from typing import Dict, Any, Optional
from pydantic import BaseModel

from ..logger_config import get_logger

logger = get_logger(__name__)

# --- Keycloak Configuration ---
# These should be set in your environment (e.g., .env file, docker-compose)
KEYCLOAK_URL = os.getenv("KEYCLOAK_URL", "http://localhost:8180")
KEYCLOAK_REALM = os.getenv("KEYCLOAK_REALM", "master")

# In-memory cache for Keycloak's public keys to avoid fetching them on every request
_public_keys_cache: Optional[Dict[str, Any]] = None

class AuthCredentials(BaseModel):
    """Pydantic model for storing validated user credentials."""
    user_id: str
    username: str
    groups: list[str]
    token: str

class JWTValidationError(Exception):
    """Custom exception for JWT validation errors."""
    pass

async def get_keycloak_public_keys() -> Dict[str, Any]:
    """
    Fetches the public keys from Keycloak's JWKS (JSON Web Key Set) endpoint.
    Results are cached in memory.
    """
    global _public_keys_cache
    if _public_keys_cache:
        return _public_keys_cache

    oidc_config_url = f"{KEYCLOAK_URL}/realms/{KEYCLOAK_REALM}/.well-known/openid-configuration"
    try:
        async with httpx.AsyncClient() as client:
            # 1. Get the OIDC configuration to find the jwks_uri
            oidc_response = await client.get(oidc_config_url)
            oidc_response.raise_for_status()
            jwks_uri = oidc_response.json()["jwks_uri"]

            # 2. Get the JSON Web Key Set (JWKS)
            jwks_response = await client.get(jwks_uri)
            jwks_response.raise_for_status()
            jwks = jwks_response.json()

            # 3. Format keys for easy lookup by key ID (kid)
            _public_keys_cache = {key['kid']: key for key in jwks['keys']}
            logger.info(f"Successfully fetched and cached {len(_public_keys_cache)} public keys from Keycloak.")
            return _public_keys_cache
    except httpx.RequestError as e:
        logger.error(f"Failed to connect to Keycloak at {oidc_config_url}: {e}")
        raise JWTValidationError("Could not connect to the authentication service.") from e
    except (httpx.HTTPStatusError, KeyError) as e:
        logger.error(f"Failed to fetch or parse Keycloak OIDC configuration: {e}")
        raise JWTValidationError("Failed to retrieve or understand authentication service configuration.") from e

async def validate_jwt(token: str) -> Dict[str, Any]:
    """
    Validates a JWT token against Keycloak's public keys.

    Args:
        token: The JWT string to validate.

    Returns:
        The decoded token payload (claims) if validation is successful.

    Raises:
        JWTValidationError: If the token is invalid, expired, or cannot be verified.
    """
    if not token:
        raise JWTValidationError("No token provided.")

    try:
        public_keys = await get_keycloak_public_keys()
        unverified_header = jwt.get_unverified_header(token)
        kid = unverified_header.get("kid")

        if not kid or kid not in public_keys:
            raise JWTValidationError("Token 'kid' (key ID) is missing or invalid.")

        rsa_key = jwk.construct(public_keys[kid])

        # The audience claim is typically the client_id in Keycloak
        # For now, we will perform a basic validation without strict audience checking.
        # In production, you should specify the expected audience.
        # audience = "your-client-id"
        
        payload = jwt.decode(
            token,
            rsa_key,
            algorithms=["RS256"],
            # audience=audience,
            # issuer=f"{KEYCLOAK_URL}/realms/{KEYCLOAK_REALM}" # Optional: verify issuer
        )
        return payload

    except JOSEError as e:
        logger.warning(f"JWT validation failed: {e}")
        raise JWTValidationError(f"Token validation error: {e}") from e
    except Exception as e:
        logger.error(f"An unexpected error occurred during JWT validation: {e}", exc_info=True)
        raise JWTValidationError("An unexpected error occurred during token validation.") from e
