import openai
import os

# --- Configuration ---
# The base URL should point to your agent_gateway's host port.
# Since the gateway's internal port 8081 is mapped to the host's port 8083,
# we use localhost:8083 when running this script from the host machine.
AGENT_GATEWAY_URL = "http://localhost:8083/v1"

# If AGENT_GATEWAY_AUTH_ENABLED is true, this should be a valid JWT from Keycloak.
# For testing purposes with auth disabled, any non-empty string will work.
API_KEY = "not-a-real-key"

# The model ID as defined in your agent_gateway's AGENT_CONFIGURATIONS
MODEL_ID = "agentic-framework/scientific-agent-v1"

# --- Create the OpenAI Client ---
# We configure the client to point to our custom agent_gateway endpoint.
client = openai.OpenAI(
    base_url=AGENT_GATEWAY_URL,
    api_key=API_KEY,
)

print(f"Attempting to connect to the Agent Gateway at: {AGENT_GATEWAY_URL}")
print(f"Using model: {MODEL_ID}")

# --- Make the API Call ---
try:
    # Call the chat completions endpoint, which is the main entry point for the agent.
    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello! What can you do?"},
        ],
        stream=False,  # Set to False to get a single, complete response object
    )

    # --- Print the Response ---
    print("\n--- Agent Gateway Response ---")
    # The response object should be similar to the standard OpenAI API response.
    # We access the content of the assistant's message from the 'choices' list.
    if response.choices:
        print(response.choices[0].message.content)
    else:
        print("The agent did not return any choices.")

    print("\n--- Full Response Object ---")
    print(response.model_dump_json(indent=2))

except openai.APIConnectionError as e:
    print("\n--- Connection Error ---")
    print(f"Failed to connect to the Agent Gateway at {AGENT_GATEWAY_URL}.")
    print("Please ensure that the Docker containers are running and that the port mapping is correct.")
    print(f"Error details: {e.__cause__}")
except openai.APIStatusError as e:
    print("\n--- API Status Error ---")
    print(f"The Agent Gateway returned an error status code: {e.status_code}")
    print(f"Response: {e.response.text}")
except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")
