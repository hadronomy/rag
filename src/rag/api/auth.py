from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader


def check_api_key(api_key: str) -> bool:
    """
    Check if the provided API key is valid.
    This function should implement the logic to verify the API key.
    """
    # Placeholder for actual API key validation logic
    # For example, you might check against a database or an environment variable
    return api_key == "valid_api_key"


def get_user_from_api_key(api_key: str):
    """
    Retrieve user information based on the provided API key.
    This function should implement the logic to fetch user details.
    """
    # Placeholder for actual user retrieval logic
    # For example, you might query a database to get user details
    return {"user_id": "123", "username": "test_user"}


api_key_header = APIKeyHeader(name="X-API-Key")


def get_user(api_key_header: str = Security(api_key_header)):
    if check_api_key(api_key_header):
        user = get_user_from_api_key(api_key_header)
        return user
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Missing or invalid API key",
    )
