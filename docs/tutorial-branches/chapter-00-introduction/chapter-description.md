# Chapter 00: Welcome to the Agentic Framework

## Getting Started

To run this chapter, navigate to the current directory and execute the lifecycle script. This script will handle all the Docker services for you.

```bash
./start-chapter-resources.sh
```

---

This chapter is your starting point for understanding and using the Agentic Discovery and Exploration Platform for Tools (ADEPT). ADEPT is a teaching tool and a practical blueprint for integrating modern AI, specifically Large Language Models (LLMs), with your scientific computing workflows. By the end of this chapter, you will have a fully functional, multi-service agentic system running on your machine and a solid understanding of the key technologies that make it work.

## Our Philosophy: A Bridge for Scientists and Engineers

This framework is more than just a collection of code; it's a teaching tool designed to foster collaboration and a shared understanding of agentic AI systems across interdisciplinary teams.

-   **For Scientific Domain Experts**: We aim to demystify the "black box" of AI. By showing how familiar scientific tools (like data analysis scripts) can be wrapped and used by an AI agent, you can better grasp the agent's capabilities and limitations. This empowers you to articulate the needs of your domain for more effective integration.

-   **For Software and Data Engineers**: This framework provides a tangible example of how to integrate specialized scientific tools into a modern, LLM-based system. It highlights robust patterns for tool wrapping, state management, and agent orchestration.

-   **Illustrative, Not Prescriptive**: Our goal is to show you *how* to connect these technologies. We focus on the technical "how-to" rather than prescribing *when* you should choose an agentic workflow over a traditional, deterministic one. That strategic decision is left to you.

## Architecture Overview

The application is composed of four main services that run in separate Docker containers and communicate with each other over a shared network. This microservice architecture makes the system modular and scalable.

```
+------------------------------------------------------------------------------------+
| Your Computer (Docker)                                                             |
|                                                                                    |
|  +-----------------+      +--------------------+      +------------------------+   |
|  |   Streamlit App | <--> |  LangChain Agent   | <--> |    MCP Tool Server     |   |
|  | (Web UI)        |      | (Reasoning Engine) |      | (FastAPI)              |   |
|  +-----------------+      +--------------------+      +------------------------+   |
|        ^   ^                      ^                            |                   |
|        |   |                      |                            v                   |
|        |   +----------------------+----------------------> [ Tools ]               |
|        |                                                     (ingest_data,         |
|        |                                                      get_sql_schema, etc.)| 
|        v                                                                           |
|  +-----------------+                                                               |
|  | JupyterLab UI   | <-----------------------------------------------------------+-|
|  | (Interactive    |                                                               |
|  |  Notebooks)     |                                                               |
|  +-----------------+                                                               |
|        |                                                                           |
|        +---------------------------------------------------------------------------+
|                                      ^                                             |
|                                      |                                             |
|  +-----------------+      +--------------------+      +------------------------+   |
|  |   Ollama Server | <--> |  ChromaDB          | <--> |    SQLite Database     |   |
|  | (LLM Provider)  |      | (Vector Store)     |      | (Relational Data)      |   |
|  +-----------------+      +--------------------+      +------------------------+   |
|                                                                                    |
+------------------------------------------------------------------------------------+
```

1.  **Streamlit App**: The primary user interface. It's a web application where you can chat with the agent and upload files. It communicates with the LangChain agent.

2.  **MCP Tool Server**: A backend service that hosts the tools the agent can use, such as `ingest_data` and `get_sql_schema`. It follows the Model Context Protocol (MCP) standard.

3.  **JupyterLab**: A secondary user interface that provides an interactive notebook environment. This is for developers and data scientists who want to interact with the MCP tools directly using Python code, bypassing the agentic workflow for more direct control.

4.  **Ollama Server**: Runs the open-source LLMs that power the agent's reasoning capabilities.

These services are supported by two persistent data stores:
-   **SQLite Database**: A traditional relational database that stores the tabular data you ingest from CSV or Excel files.
-   **ChromaDB**: A vector store that acts as the agent's long-term memory, storing chat history, file upload records, and notes.

## Key Technologies in this Framework

-   **[Docker]**: Used to containerize each service, ensuring a consistent and reproducible environment.
-   **[LangChain]**: The core agentic framework used to build the reasoning and tool-use logic. We use its `StateGraph` to create a sophisticated, conditional control flow that prevents the agent from getting stuck in loops.
-   **[FastMCP]**: A Python library for creating MCP-compliant tool servers, making our tools discoverable and usable by the agent.
-   **[Pandas]**: A powerful data analysis library used within the `ingest_data` tool to robustly parse CSV and Excel files, gracefully handling bad rows and mixed data types.
-   **[ChromaDB]**: A vector database that provides long-term memory for the agent.
-   **[Streamlit]**: A Python library for creating interactive web applications, used here for the main user interface.
-   **[JupyterLab]**: A web-based interactive development environment for notebooks, code, and data.

## References
- [Docker]: https://www.docker.com/
- [LangChain]: https://www.langchain.com/
- [FastMCP]: https://github.com/gofastmcp/fastmcp
- [Pandas]: https://pandas.pydata.org/
- [ChromaDB]: https://www.trychroma.com/
- [Streamlit]: https://streamlit.io/
- [JupyterLab]: https://jupyter.org/

## How to Use the Application

The application provides two primary interfaces for interacting with your data: the Streamlit web app and the JupyterLab notebook environment.

### Using the Streamlit App

1.  **Access the App**: Open your web browser and navigate to `https://localhost:8501`.
2.  **Bypass Security Warning**: You will see a security warning because the app uses a self-signed certificate. It is safe to proceed. Click `Advanced` and then `Proceed to localhost (unsafe)`.
3.  **Chat with the Agent**: Use the chat input at the bottom of the page to ask questions about your data. The agent will use its tools to answer.
4.  **Upload and Ingest Files**: Use the sidebar to upload one or more (up to 5) CSV or Excel files. Click the "Ingest Files" button to load them into the database.

### Using the JupyterLab UI

The JupyterLab environment is for developers and data scientists who want to interact with the system's tools directly, without going through the conversational agent.

1.  **Get the Access Token**: The JupyterLab server is protected by a security token. You need to get this token from the Docker logs the first time you run the service.
    -   Open a terminal and run: `docker compose logs jupyterlab`
    -   Look for a line that looks like this:
        ```
        http://127.0.0.1:8888/lab?token=a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4
        ```
    -   The long string of characters after `?token=` is your access token. Copy it.

2.  **Access JupyterLab**: Open a new browser tab and navigate to `http://localhost:8888`.

3.  **Enter the Token**: Paste the token you copied into the password field and click "Log in".

4.  **Open the Example Notebook**: 
    -   In the file browser on the left, you will see a `notebooks` directory.
    -   Double-click `mcp_interaction_example.ipynb` to open it.

5.  **Run the Notebook Cells**: This notebook provides a complete, runnable example of how to use the `fastmcp` client library to call the tools on the MCP server directly. You can run the cells one by one to see how to:
    -   List all available tools.
    -   Ingest the sample real-estate data.
    -   Retrieve the database schema.
    -   Execute a direct SQL query.

## Running the Application with Docker

This is the easiest and most reliable way to run the application.

### 1. Configure Environment Variables

Before you start, you must create a `.env` file to hold your configuration.

-   **Create the `.env` file**: Copy the example file.
    ```bash
    cp .env.example .env
    ```
-   **Set User and Group IDs**: This is critical for JupyterLab to have the correct file permissions.
    -   On macOS or Linux, find your user ID by running `id -u` and your group ID by running `id -g`.
    -   Open the `.env` file and set the `UID` and `GID` variables to these values.
-   **(Optional) Set API Keys**: If you want to use proprietary models, add your API keys for services like Azure OpenAI or Google Vertex AI.

### 2. Build and Run the Containers

-   **From the Command Line**:
    1.  Open a terminal and navigate to the project's root directory.
    2.  Build the images and start the services:
        ```bash
        docker compose up --build -d
        ```
        -   `--build`: Rebuilds the images if the code or Dockerfiles have changed.
        -   `-d`: Runs the containers in the background.

### 3. Monitor the Services

-   **View Logs**: To see the real-time output from all services, run:
    ```bash
    docker compose logs -f
    ```
-   **Stop the Services**: To stop all running containers, run:
    ```bash
    docker compose down
    ```


### How It Works: The Core Components

This introductory chapter sets up the essential services for a functioning agentic system. Here’s how they interact:

**Example User Query:** "What time is it, and what is 5*5?"

1.  **User Interface (Streamlit)**: You interact with the system through the Streamlit web app. When you type the query, the app sends it to the LangChain Agent.

2.  **Reasoning Engine (LangChain Agent)**: The agent, powered by an LLM (via Ollama), receives your query. It recognizes that the query has two distinct parts that require two different tools.

3.  **First Tool Call**: The agent first calls the `get_current_datetime` tool on the MCP server. The server executes the function and returns the current time to the agent.

4.  **Second Tool Call**: The agent then calls the `perform_calculation` tool with the expression `5*5`. The server evaluates the expression and returns the result (25) to the agent.

5.  **Response Generation**: The agent synthesizes the results from both tool calls into a single, human-readable sentence (e.g., "The current time is... and 5*5 is 25.") and sends this final answer back to the Streamlit UI.

6.  **Memory (ChromaDB)**: Throughout this process, the conversation is stored in the ChromaDB vector store, allowing the agent to maintain context for future interactions.

## Additional Resources

### Key Concepts

#### Agentic AI
-   **Courses**:
    -   [Agentic AI: A Primer For Leaders (Coursera)](https://www.coursera.org/learn/agentic-ai?)
    -   [The Complete Agentic AI Engineering Course (Udemy)](https://www.udemy.com/course/the-complete-agentic-ai-engineering-course/)
-   **Tutorials**:
    -   [Building Agentic AI Free Course (Krish Naik Academy)](https://www.youtube.com/watch?v=qR3HWsMFfZA)
    -   [The Agentic AI Handbook](https://www.freecodecamp.org/news/the-agentic-ai-handbook/)

#### Large Language Models (LLMs)
-   **Courses**:
    -   [Introduction to Large Language Models by Google](https://www.classcentral.com/course/google-introduction-to-large-language-models-101203)
    -   [Generative AI for Beginners by Microsoft](https://microsoft.github.io/generative-ai-for-beginners/)
-   **Tutorials**:
    -   [Building LLMs from the Ground Up: A 3-hour Coding Workshop](https://www.youtube.com/watch?v=UU1WVnMk4E8)

### Key Technologies

#### Docker
-   **Courses**:
    -   [Docker & Kubernetes: The Practical Guide (Udemy)](https://www.udemy.com/course/docker-kubernetes-the-practical-guide/)
    -   [Introduction to Containers w/ Docker, Kubernetes & OpenShift (Coursera)](https://www.coursera.org/learn/ibm-containers-docker-kubernetes-openshift?)
-   **Tutorials**:
    -   [Docker's Official Tutorial](https://www.docker.com/101-tutorial/)
    -   [Free Docker Course by DevOps Directive](https://www.youtube.com/watch?v=3c-iBn73dDE)

#### LangChain
-   **Courses**:
    -   [LangChain for LLM Application Development (DeepLearning.AI)](https://www.deeplearning.ai/short-courses/langchain-for-llm-application-development/)
    -   [LangChain: Chat with Your Data (DeepLearning.AI)](https://www.deeplearning.ai/short-courses/langchain-chat-with-your-data/)
-   **Tutorials**:
    -   [Official LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)

#### FastMCP
-   **Courses**:
    -   [The Complete MCP Course: From Zero to Deployment (YouTube)](https://www.youtube.com/watch?v=kQmXtrmQ5Zg)
-   **Tutorials**:
    -   [Official FastMCP Documentation](https://gofastmcp.com/)
    -   [Building Your First FastMCP Server: A Complete Guide](https://medium.com/p/b2a176eaa631)
