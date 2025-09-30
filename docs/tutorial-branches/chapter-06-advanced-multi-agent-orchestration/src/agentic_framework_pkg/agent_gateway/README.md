# OpenWebUI App

This submodule provides a backend for running the scientific workflow agent with the OpenWebUI frontend.

## Running the OpenWebUI Interface

To run the OpenWebUI interface, you need to use both the main `docker-compose.yaml` file and the `docker-compose-openwebui.yaml` file.

Use the following command to build and start all the services. The agentic-framework agent_gateway module will serve as the backend to the openwebui front-end.

```bash
COMPOSE_BAKE=true docker compose -f docker-compose.yaml -f docker-compose-openwebui.yaml build
docker compose -f docker-compose.yaml -f docker-compose-openwebui.yaml up 
```


Launch the OpenWebUI interface that will refer to the agentic-framework's compatible backend. Note that port 8083 is the host port where the openwebui backend is served.

```bash
docker login ghcr.io # You'll need to setup a PAT, if you haven't already, see below

# For testing, run from the root of the project folder
docker run -p 8902:8080 -v open-webui:`pwd`/data --name open-webui --restart always ghcr.io/open-webui/open-webui:main
```

Once the services are running, you can access the OpenWebUI interface at [http://localhost:8902](http://localhost:8902).

You will then add the openwebui backend (http://localhost:8083) to the set of the UI's connections.


## To setup a GitHub PAT (personal access token):

1. Logout and Re-login to Docker
First, log out of Docker to clear any potentially stale or incorrect credentials. Then, log in again, ensuring you use the correct credentials for ghcr.io.

```bash
docker logout
```

After logging out, you'll need to log in again. When logging into ghcr.io, you use your GitHub username and a GitHub Personal Access Token (PAT) as the password. You cannot use your regular GitHub password directly for docker login to ghcr.io.


### Steps to create a GitHub Personal Access Token (PAT):

Go to your GitHub profile settings.

Navigate to Developer settings -> Personal access tokens -> Tokens (classic).

Click "Generate new token" -> "Generate new token (classic)".

Give your token a descriptive name (e.g., "Docker GHCR Access").

Under "Select scopes", ensure you select at least read:packages (to pull images) and potentially write:packages if you ever plan to push images. For just pulling, read:packages should suffice.

Click "Generate token". Copy the token immediately, as you won't be able to see it again.

Once you have your PAT, log in to Docker again, specifically for ghcr.io:

```bash
docker login ghcr.io
```

When prompted for "Username", enter your GitHub username. When prompted for "Password", paste your GitHub Personal Access Token (PAT).

