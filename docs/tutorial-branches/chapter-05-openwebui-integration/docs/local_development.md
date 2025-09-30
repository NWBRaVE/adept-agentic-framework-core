# Local Development Guide

This guide will walk you through setting up your local development environment for the Agentic Framework. We will cover setting up prerequisites, cloning the repository, and running the application stack using Docker Compose and Helm on a local Kubernetes cluster (via Docker Desktop).

## Part 1: Prerequisites

This section covers setting up your development environment on macOS and Windows.

### Core Tools

Before you begin, ensure you have the following installed:

1.  **Git**: For version control.
2.  **Python**: Version 3.11 or higher.
3.  **Docker Desktop**: For containerizing and running the application, and for its built-in Kubernetes cluster.
4.  **Visual Studio Code (VS Code)**: Recommended code editor.

### macOS Setup

1.  **Homebrew**: If you don't have Homebrew, install it by running the command from brew.sh.
2.  **Install Tools**:
    ```bash
    brew install git python@3.11
    ```
3.  **Docker Desktop**: Download and install Docker Desktop for Mac from the Docker website.
4.  **VS Code**: Download and install VS Code from code.visualstudio.com.
    *   **Recommended Extensions**: `Python` (Microsoft), `Docker` (Microsoft).

### Windows Setup

1.  **Windows Subsystem for Linux (WSL)**: This is highly recommended for a smoother development experience.
    *   Open PowerShell as Administrator and run: `wsl --install`. This will install Ubuntu by default.
2.  **Install Tools (inside WSL)**:
    ```bash
    sudo apt update
    sudo apt install git python3.11 python3.11-venv python3-pip
    ```
3.  **Docker Desktop**: Download and install Docker Desktop for Windows. Ensure it's configured to use the **WSL 2 backend**.
4.  **VS Code**: Download and install VS Code.
    *   **Recommended Extensions**: `Python` (Microsoft), `Docker` (Microsoft), `WSL` (Microsoft).

## Part 2: Project Setup

1.  **Clone the Repository**:
    *   If using WSL, open your WSL terminal.
    *   Clone the repository:
        ```bash
        git clone <your-repo-url>
        cd agentic-framework
        ```

2.  **Open in VS Code**:
    *   **If using WSL**: In your WSL terminal, inside the project directory, type:
        ```bash
        code .
        ```
    *   **If not using WSL**: Open VS Code and use "File > Open Folder..." to open the cloned project directory.

3.  **Set up `.env` file**:
    *   The project uses a `.env` file for environment variables and secrets.
    *   Copy the example environment file to `.env`:
        ```bash
        cp .env.example .env
        ```
    *   Edit the `.env` file and fill in necessary API keys (e.g., `OPENAI_API_KEY`, `AZURE_API_KEY`) and configurations.

## Part 3: Running the Application

There are two primary ways to run the application locally: using Docker Compose for a simple multi-container setup, or deploying to the local Kubernetes cluster in Docker Desktop for a more production-like environment.

### Option A: Running with Docker Compose

This is the simplest method for getting all services up and running.

1.  **Ensure Docker Desktop is running.**
2.  **Build and Run**:
    *   From the project root directory, build the Docker images:
        ```bash
        docker compose build
        ```
    *   Launch the containers in detached mode:
        ```bash
        docker compose up -d
        ```
3.  **Access the Application**:
    *   The Streamlit UI will be available at `http://localhost:8501`.
4.  **View Logs**:
    ```bash
    docker compose logs -f # Follow logs for all services
    docker compose logs -f mcp_server # Follow logs for a specific service
    ```
5.  **Stop the Application**:
    ```bash
    docker compose down
    ```

### Option B: Deploying to Local Kubernetes (Docker Desktop)

This method uses Helm to deploy your application to the Kubernetes cluster that comes with Docker Desktop. This is a great way to test your Kubernetes configuration locally.

1.  **Enable Kubernetes in Docker Desktop**:
    *   Open Docker Desktop Settings.
    *   Go to the **Kubernetes** tab.
    *   Check the **Enable Kubernetes** box and click "Apply & Restart".

2.  **Build Local Docker Images**:
    *   Kubernetes needs access to the container images. For local development, we build them directly into Docker Desktop's local registry.
    *   Run this command to build all service images:
        ```bash
        docker compose build
        ```

3.  **Configure Helm `values.yaml` for Local Deployment**:
    *   Open `infra/helm/agentic-framework/values.yaml`.
    *   Ensure the following settings are configured for local development:
        ```yaml
        global:
          image:
            pullPolicy: Never # Crucial for using locally built images
            tag: "latest"

        sandbox_mcp_server:
          # For local Docker Desktop, the 'docker' backend is often simpler.
          # Ensure Docker Desktop settings allow socket sharing.
          backend: "docker"
        ```

4.  **Deploy with Helm**:
    *   Open your terminal and run the following commands to install the application as a Helm release.
        ```bash
        # Create the namespace if it doesn't exist
        kubectl create namespace agentic-framework

        # Install or upgrade the Helm release
        helm upgrade --install my-agentic-release ./infra/helm/agentic-framework/ --namespace agentic-framework
        ```

5.  **Access the Application**:
    *   The Helm chart's `NOTES.txt` provides the easiest way to connect. After the `helm upgrade` command finishes, it will print instructions. The recommended command is:
        ```bash
        kubectl port-forward --namespace agentic-framework svc/my-agentic-release-agentic-framework-streamlit-app 8501:8501
        ```
    *   Now, open your browser and go to `http://localhost:8501`.

6.  **View Pods and Logs**:
    *   Check the status of your pods:
        ```bash
        kubectl get pods --namespace agentic-framework
        ```
    *   Stream logs from a specific pod:
        ```bash
        kubectl logs -f <pod-name> --namespace agentic-framework
        ```

7.  **Uninstall the Application**:
    *   To remove all Kubernetes resources created by Helm:
        ```bash
        helm uninstall my-agentic-release --namespace agentic-framework
        ```