# Chapter 06: Advanced Multi-Agent Orchestration and Vector Database

## Getting Started

To run this chapter, navigate to the current directory and execute the lifecycle script. This script will handle all the Docker services for you.

```bash
./start-chapter-resources.sh
```

---

This chapter significantly enhances the framework's multi-agent and data management capabilities. It introduces a dedicated vector database service and provides two distinct modes for orchestrating agent workflows: a static, plan-based approach and a dynamic, graph-based system. It also introduces a new Agent Gateway service to provide a unified, secure, and protocol-agnostic entry point for all agent interactions.

## Key Additions

- **Agent Gateway Service**: A new, central `agent_gateway` service is introduced. It provides a unified, OpenAI-compatible API for all clients, including OpenWebUI and n8n. It manages tool registration, authentication, and authorization, and dynamically provisions the `ScientificWorkflowAgent` with the appropriate tools for each user session.

- **Dedicated Vector Database**: The framework now includes a standalone ChromaDB server, defined in `docker-compose.yaml` using the official `chromadb/chroma` image. This service provides a persistent, scalable, and centralized vector store for all RAG (Retrieval-Augmented Generation) and embedding-based tools, replacing the previous file-based ChromaDB implementation.

- **Enhanced Multi-Agent Orchestration**: The `multi_agent_tool` has been significantly upgraded to support more robust and intelligent workflows.
    - **Router Mode with "Plan and Review"**: The traditional plan-based approach has been enhanced with a collaborative planning cycle. The agent can now generate a plan and have it automatically reviewed and refined by the expert agent team *before* presenting it to the user for approval. This leads to more robust and reliable plans.
    - **Graph Mode with Loop Prevention**: The dynamic, `langgraph`-based supervisor now includes a mechanism to track agent turns. This allows it to detect and prevent infinite loops, making it more resilient for complex, non-linear tasks.

- **OpenWebUI MCP Backend**: A new `openwebui_mcp_backend` service is introduced, defined in `Dockerfile.openwebui_mcp_backend`. This service provides an OpenAI-compatible API that allows the multi-agent system to be used as a backend model within OpenWebUI, enabling a rich chat experience for orchestrating complex, multi-agent workflows.

- **n8n Workflow Automation Integration**: The framework now includes a containerized n8n service, defined in `docker-compose-n8n.yaml`. This allows you to use the `ScientificWorkflowAgent` as an intelligent node within n8n workflows, automating complex tasks by connecting it to hundreds of other applications and services.

## Consolidated Architecture and Implementation Details

This section provides a consolidated view of the Agent Gateway's architecture, incorporating the latest docstrings and implementation details.

### Core Components and Flow

The Agent Gateway is composed of three main components that work together to dynamically provision and execute tools for the `ScientificWorkflowAgent`.

#### Multi-Session and Multi-Tenant Support

The Agent Gateway is designed to support both multi-session and multi-tenant environments, ensuring that multiple users can interact with the system concurrently and securely, each with their own isolated conversations and authorized tools.

**Multi-Session Support:**
- **Conversation ID (`conversation_id`):** Each chat interaction is assigned a unique `conversation_id` (or a new one is generated if not provided). This ID serves as the primary key for managing distinct chat sessions within the Agent Gateway.
- **Session Management (`sessions` dictionary):** The `app.py` module maintains a global `sessions` dictionary, which stores individual `ScientificWorkflowAgent` instances, keyed by their respective `conversation_id`. This ensures that each unique conversation has its own dedicated agent instance and state.
- **MCP Session ID (`mcp_session_id`):** The `conversation_id` is passed as the `mcp_session_id` to the `ScientificWorkflowAgent` constructor. This `mcp_session_id` is then propagated to all `MCPToolWrapper` instances, ensuring that stateful MCP tool calls (e.g., storing notes, querying user-specific data) are correctly associated with the specific conversation.
- **Redis Checkpointing:** When persistence is enabled (`USE_CHECKPOINTING=true`), the `mcp_session_id` is used as the `thread_id` for the `RedisSaver` (checkpointer). This allows each conversation's state to be independently persisted and retrieved from Redis, enabling long-running conversations and recovery from interruptions.

**Multi-Tenant Support:**
- **JWT-Based Authentication:** The `app.py` module leverages JWT (JSON Web Token) for authentication. If authentication is enabled, incoming JWTs are validated, and the `user_id` and `groups` of the authenticated user are extracted.
- **Dynamic Tool Authorization (ACLs):** The `ExternalToolManager` plays a crucial role in multi-tenancy. Tools registered with the gateway can include an Access Control List (ACL) that specifies `global` access, specific `users`, or `groups` that are authorized to use the tool.
- **Authorized Tool Provisioning:** During the `ScientificWorkflowAgent` instantiation, the `tool_manager.get_authorized_tools()` method is called with the authenticated `user_id` and `user_groups`. This method filters the available tools based on the defined ACLs, ensuring that each user (tenant) is only provisioned with the tools they are explicitly authorized to use. This prevents unauthorized access to sensitive tools or data.

This robust design allows the Agent Gateway to serve multiple users concurrently, maintaining session isolation and enforcing fine-grained access control over available tools.

#### `app.py`: The FastAPI Application

The `app.py` module is the main entry point for the Agent Gateway. It defines the API endpoints, including the crucial `/v1/chat/completions` endpoint, which orchestrates the entire agent lifecycle.

**`/v1/chat/completions` Endpoint Flow:**

```
User Request ----------------> Agent Gateway (/v1/chat/completions)
                                     |
           +-------------------------+-------------------------+
           | 1. Authenticate User                            |
           |                                                 |
           | 2. Call ExternalToolManager.get_authorized_tools() |
           |    (Queries DB, filters by ACL)                 |
           |                                                 |
           | 3. Create LangChain Tool objects (HTTP or stdio) |
           |                                                 |
           | 4. Instantiate ScientificWorkflowAgent          |
           |    with the list of authorized Tool objects     |
           +-------------------------+-------------------------+
                                     |
                                     V
ScientificWorkflowAgent (now has its tools) <-----> LangGraph Runtime
           ^                                                 |
           | (Tool Result)                                   V (LLM decides to use a tool)
           |                                                 |
           +------------------ Tool._arun() is called <-------+
                                     |
                 +-------------------+-------------------+
                 |                                       |
 V (HTTP Request)                                V (stdio Subprocess)
                 |                                       |
         External HTTP Service                  Docker Container / Local Command
```

1.  **Authentication and Authorization:**
    - The endpoint first uses the `get_current_user` dependency to validate the incoming JWT (if authentication is enabled). This step is crucial for identifying the user and their group memberships, which are used to enforce tool access control.

2.  **Session and Agent Scoping:**
    - It checks for an existing agent instance tied to the `conversation_id`.
    - If no agent exists for the session, or if the requested `model` has changed, it begins the process of creating a new `ScientificWorkflowAgent`.

3.  **Dynamic Tool Provisioning:**
    - This is the core of the dynamic agent architecture. Before creating the agent, the gateway gathers all the tools the agent will be allowed to use.
    - It starts with any static tools defined in `AGENT_CONFIGURATIONS` (e.g., the `echo_message_tool`).
    - It then calls `tool_manager.get_authorized_tools()`, passing the user's ID and groups. The `ExternalToolManager` queries its database for all registered external tools (both HTTP and stdio types) and filters them based on the user's permissions defined in each tool's Access Control List (ACL).
    - For each authorized tool configuration, the `ExternalToolManager` creates a fully-formed, LangChain-compatible `Tool` object.

4.  **Agent Instantiation:**
    - A new `ScientificWorkflowAgent` instance is created. The list of authorized `Tool` objects is passed directly to its constructor.
    - This means the agent instance is "born" with a toolset tailored specifically for the current user and session. It does not need to know about the `ExternalToolManager` or the tool registration process; it only knows the tools it was given.

5.  **Execution and Streaming:**
    - The user's query is passed to the agent's `arun` method.
    - The agent, managed by a LangGraph runtime, uses the LLM to reason about the user's request and its available tools.
    - When the LLM decides to use a tool, LangGraph invokes the corresponding `Tool` object. The tool's internal logic (e.g., making an HTTP request or executing a `docker run` command) is completely abstracted away from the agent.
    - The final response from the agent is streamed back to the client in the OpenAI-compatible Server-Sent Events (SSE) format.

**`/v1/tools/register` Endpoint:**

This endpoint allows administrators to dynamically register new tools with the gateway. The request body is a JSON object that defines the tool's properties, including its invocation method and access control list.

**Example Payloads:**

*   **HTTP Tool Registration:**

    ```json
    {
      "name": "protein_sequence_fetcher",
      "description": "Fetches protein sequence data from an external API.",
      "args_schema": {
        "uniprot_id": {
          "type": "string",
          "description": "The UniProt accession ID (e.g., 'P0DTD1')."
        }
      },
      "invocation": {
        "protocol": "http",
        "connection": {
          "url": "https://api.example.com/proteins/fetch",
          "method": "GET",
          "headers": {
            "X-API-Key": "your-secret-api-key"
          }
        }
      },
      "acl": {
        "global": false,
        "groups": ["bioinformatics_team"]
      }
    }
    ```

*   **Docker `run` stdio Tool Registration:**

    ```json
    {
      "name": "my_docker_tool",
      "description": "A tool that runs a command inside a Docker container.",
      "args_schema": {
        "input_arg": {
          "type": "string",
          "description": "An input argument for the tool."
        }
      },
      "invocation": {
        "protocol": "stdio",
        "connection": {
          "command": ["docker", "run", "--rm", "-i", "my_docker_image:latest", "my_tool_command"]
        }
      },
      "acl": {
        "global": true
      }
    }
    ```

*   **Docker MCP Toolkit `gateway run` stdio Tool Registration:**

    ```json
    {
      "name": "my_mcp_toolkit_tool",
      "description": "A tool that uses the Docker MCP toolkit to run a gateway command.",
      "args_schema": {
        "mcp_tool_name": {
          "type": "string",
          "description": "The name of the MCP tool to invoke."
        },
        "mcp_tool_args": {
          "type": "object",
          "description": "The arguments for the MCP tool."
        }
      },
      "invocation": {
        "protocol": "stdio",
        "connection": {
          "command": ["docker", "mcp", "gateway", "run"]
        }
      },
      "acl": {
        "global": true
      }
    }
    ```

#### `external_tool_manager.py`: The Tool Registry and Factory

The `ExternalToolManager` class is the central registry and factory for all dynamically registered tools. It abstracts away the complexities of tool invocation, allowing the agent to interact with any tool through a unified, LangChain-compatible interface.

**Key Responsibilities:**

1.  **Persistence:**
    -   Tool configurations, including their invocation details (URL, command, etc.) and Access Control Lists (ACLs), are persisted in a ChromaDB collection.
    -   This ensures that registered tools are not lost when the gateway restarts.

2.  **Protocol-Agnostic Factory:**
    -   The manager uses a dictionary of factory functions (`protocol_factories`) to create `langchain.tools.Tool` objects from their stored configurations.
    -   It supports multiple communication protocols (e.g., `http`, `stdio`). When a tool is to be created, the manager looks at the `protocol` specified in its configuration and calls the corresponding factory function (e.g., `_create_http_tool`, `_create_stdio_tool_from_config`).
    -   This design is highly extensible; adding support for a new protocol simply requires adding a new factory function to the dictionary.

3.  **Dynamic Tool Loading and Caching:**
    -   On startup, the manager loads all tool configurations from the database and creates the corresponding `Tool` objects, caching them in memory in the `self.registered_tools` dictionary for fast access.
    -   When a new tool is registered via the `/v1/tools/register` endpoint, it is added to both the database and the in-memory cache.

4.  **Authorization and Access Control:**
    -   The `get_authorized_tools` method is the critical link between the tool registry and the agent. It takes a user's ID and group memberships (from their JWT) and filters the master list of cached tools.
    -   It returns only the `Tool` objects that the user is permitted to use, based on the `acl` metadata stored with each tool's configuration.
    -   This ensures that when the `ScientificWorkflowAgent` is instantiated, it is only provisioned with the tools it is authorized to access.

#### `stdio_tool_wrapper.py`: The stdio Execution Engine

This module provides the core functionality for creating and executing tools that communicate over standard I/O (stdio). It defines a generic wrapper that can turn any command-line executable that reads JSON from stdin and writes JSON to stdout into a LangChain-compatible `Tool`.

**Architectural Role:**

This wrapper is the key component that enables the "Lightweight Local Execution" pattern. It decouples the Agent Gateway from the specific implementation of a tool, allowing it to execute tools written in any language, as long as they adhere to the simple stdio JSON contract.

It is used by the `ExternalToolManager`'s `_create_stdio_tool_from_config` factory to create tool instances for any registered tool that specifies `"protocol": "stdio"`.

**How It Works:**

1.  **`create_stdio_tool` (Factory Function):**
    -   This function receives the tool's configuration (name, description, command, and Pydantic schema for arguments).
    -   It dynamically creates a coroutine (`tool_func`) that will be the core execution logic for the LangChain `Tool`.
    -   It returns a `langchain.tools.Tool` instance, passing the `tool_func` as its `coroutine`.

2.  **`_run_stdio_tool` (Execution Logic):**
    -   This async function is called by the `tool_func` when the agent decides to use the tool.
    -   It takes the command to execute (e.g., `["docker", "run", "my-image"]`) and the arguments provided by the LLM.
    -   It uses Python's `asyncio.create_subprocess_exec` to run the command as a separate process.
    -   It serializes the input arguments into a JSON string and writes them to the subprocess's `stdin`.
    -   It captures the `stdout` and `stderr` from the subprocess.
    -   If the process exits with an error, it returns a formatted error message.
    -   If successful, it returns the JSON string received from the tool's `stdout`.

This mechanism allows the Agent Gateway to seamlessly invoke a wide variety of external processes, from simple Python scripts to complex Docker containers, as if they were native Python tools.

## Design Discussion: The Agent as a Universal Backend

A key architectural theme in this chapter is the use of a standardized, OpenAI-compatible API (served by the `agent_gateway`) to make the `ScientificWorkflowAgent` universally accessible. This single API endpoint allows diverse applications—a chat UI like OpenWebUI and a workflow engine like n8n—to seamlessly use the agent as a powerful, tool-enabled "brain" without requiring custom integration code. This demonstrates a powerful pattern for building modular, interoperable AI systems.

## Design Discussion: Dual-Mode Multi-Agent Orchestration

The framework supports two distinct orchestration patterns for managing agent teams, providing the flexibility to choose the best strategy for the task at hand. The choice is made explicitly when the agent calls the `create_multi_agent_session` tool.

| Execution Mode      | How It's Chosen                                                                                                                            | Governed by System Prompt?                                                                                             |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------- |
| **Router (Simple)** | The agent reasons that the task is simple and calls `GeneratePlanForMultiAgentTask`.                                                       | **Yes**, the prompt guides this choice.                                                                                |
| **Router (Enhanced)** | The agent reasons that the task is complex and calls `GenerateAndReviewPlanForMultiAgentTask`.                                             | **Yes**, the prompt guides this choice.                                                                                |
| **Graph-Based**     | The `CreateMultiAgentSession` tool is called with the explicit parameter `execution_mode='graph'`. The agent then uses a different tool (`execute_task_in_graph_session`) to interact with it. | **No**, the prompt intentionally omits instructions for this mode to ensure the agent follows the router workflow. |

### Router Mode (Default)
This is a deterministic, plan-based approach suitable for tasks with a clear sequence. The main `ScientificWorkflowAgent` is guided by its system prompt to follow a strict lifecycle:
1.  **Create:** The agent calls `create_multi_agent_session` with `execution_mode='router'`.
2.  **Plan:** The agent uses its reasoning to select a planning tool.
    -   **Simple Plan:** For straightforward tasks, it calls `generate_plan_for_multi_agent_task`.
    -   **Enhanced Plan:** For complex tasks, it calls `generate_and_review_plan_for_multi_agent_task`, which involves an internal review cycle by the agent team before the plan is finalized.
3.  **Approve:** The agent presents the plan to the user for approval.
4.  **Execute:** Upon approval, the agent calls `execute_approved_plan_in_session`. The Supervisor agent then delegates each step of the plan to the appropriate worker.

### Graph Mode (Advanced)
This is a more flexible, dynamic approach for non-linear problems where the path to a solution is not known in advance.
1.  **Create:** The agent must be explicitly instructed to call `create_multi_agent_session` with the parameter `execution_mode='graph'`.
2.  **Execute:** The agent calls `execute_task_in_graph_session`. The `supervisor_graph` then dynamically routes tasks between worker agents based on the evolving state of the problem, continuing until a solution is reached.

The main agent's system prompt is intentionally focused on the `router` mode to ensure robust and predictable behavior for the most common workflows.

## Design Discussion: The "Split-Stream" Strategy for Tool Outputs

A significant architectural improvement in this chapter is the implementation of a "split-stream" strategy for handling tool outputs within `langchain_agent.py`. This design addresses a critical challenge in advanced agentic workflows: managing the LLM's limited context window while preserving full data fidelity for UI and programmatic use.

### The Challenge

When a tool, such as the `ExecuteCode` tool, generates a large data payload (e.g., a base64-encoded plot), passing the entire payload back to the LLM in a `ToolMessage` can quickly consume its context window. This is especially problematic for models with smaller context limits and can degrade the agent's performance or cause it to fail. However, the application (like the Streamlit UI) still needs the full data to render the plot correctly.

### The Solution

The "split-stream" strategy solves this by creating two versions of a tool's output:

1.  **Lightweight (Sanitized) Output for the LLM:** A sanitized version of the tool's result is generated where large data fields (like `content_base64`) are replaced with a placeholder (e.g., `<PNG data omitted, use plot_url>`). This lightweight JSON string is what gets sent to the LLM, ensuring its context is not polluted with large, unnecessary data.

2.  **Full Programmatic Output:** The original, unmodified output from the tool, including the complete `content_base64` data, is stored in a separate field (`full_tool_outputs`) within the agent's state.

This approach was implemented by replacing the high-level `create_react_agent` utility with a custom `StateGraph`. This provided the granular control needed to define a custom tool node that intercepts the output, performs the split, and directs each stream to its proper destination.

This design choice offers the best of both worlds:
- **Efficient LLM Interaction:** The agent's core reasoning process remains fast and efficient.
- **Full Data Fidelity:** The UI and other downstream components have access to the complete, rich data they need to function correctly.

## Design Discussion: Static Tool Factory for Agent Capabilities

This framework uses a "Static Factory" pattern to equip the LangChain agent with its tools, a design choice detailed in `mcp_langchain_tools.py`.

### The Approach

Instead of having the agent dynamically discover tools at runtime (e.g., by calling a `list-tools` endpoint), the framework uses a series of explicit factory functions (`get_mcp_*_tool_langchain`). The main `ScientificWorkflowAgent` calls these factories during its initialization to build a static, hardcoded list of all the tools it can use.

### Advantages vs. Disadvantages

-   **Advantage - Clarity, Security, and Reliability:** This approach provides a clear, auditable "allowlist" of the agent's capabilities. It's more secure because the agent cannot discover and call a potentially sensitive tool that wasn't explicitly assigned to it. It also improves the reliability of the LLM's reasoning, as the agent's system prompt can be more accurate and detailed.

-   **Disadvantage - Less Dynamic:** The primary trade-off is that adding a new tool to an MCP server requires a manual code change. A developer must add a corresponding factory function to `mcp_langchain_tools.py` and update the agent's tool list.

This static approach was chosen because it prioritizes security and reliability, which is a significant architectural advantage in a complex system with multiple specialized tool servers.

## Key Technologies

- [ChromaDB](https://www.trychroma.com/): An open-source embedding database for building AI applications with memory.
- [LangGraph](https://langchain-ai.github.io/langgraph/): A library for building stateful, multi-agent applications with LLMs.
- [Multi-Agent Systems](https://en.wikipedia.org/wiki/Multi-agent_system): A computerized system composed of multiple interacting intelligent agents.

## Tutorial: Dynamically Registering and Using a New Tool

This tutorial demonstrates the power of the Agent Gateway by walking through the process of adding a new command-line tool to the agent's capabilities *without restarting any services*.

**Our Goal:** We will create a simple tool named `get_system_information` that reports the OS and architecture of the container it's running in. This is a `stdio` tool, meaning it communicates via standard input and output.

### Prerequisites

Ensure your Docker environment is running on your remote server with all the necessary services. You should have executed the following commands from the `tutorial-branches/chapter-06-advanced-multi-agent-orchestration` directory on that server:

```bash
# Build all the service images
COMPOSE_BAKE=true docker compose -f docker-compose.yaml -f docker-compose-openwebui.yaml build

# Start all the services in the background
docker compose -f docker-compose.yaml -f docker-compose-openwebui.yaml up -d
```
Also, ensure you have an SSH session with port forwarding for both OpenWebUI (`8902`) and the Agent Gateway (`8083`):
```bash
ssh -L 8902:localhost:8902 -L 8083:localhost:8083 user@remote-server
```

### Step 1: Create the Tool's Python Script

First, we need the script that the gateway will execute. This script must read JSON from `stdin` and write its result as JSON to `stdout`. This file has already been created for you at `src/agentic_framework_pkg/agent_gateway/tools/sys_info_tool.py`.

### Step 2: Create the Tool's JSON Registration File

Next, we need the JSON file that describes our new tool to the agent gateway. This file tells the gateway what the tool is called, what it does, and how to run it. This file has also been created for you at `src/agentic_framework_pkg/agent_gateway/cli/examples/stdio_sys_info_tool.json`.

### Step 3: Register the New Tool

This step involves sending the JSON configuration to the running `agent_gateway`'s `/v1/tools/register` endpoint. Since the gateway is on your remote server, you must run the command from your local machine where the port `8083` is being forwarded.

Open a **new local terminal** (not your SSH session) and run the following `curl` command. This command reads the content of your JSON file and sends it as the body of a `POST` request.

```bash
curl -X POST http://localhost:8083/v1/tools/register \
-H "Content-Type: application/json" \
-d @docs/tutorial-branches/chapter-06-advanced-multi-agent-orchestration/src/agentic_framework_pkg/agent_gateway/cli/examples/stdio_sys_info_tool.json
```

You should see a success message like:
`{"status":"success","message":"Tool 'get_system_information' registered successfully..."}`

### Step 4: Verify the Registration

Now, let's verify that the agent gateway recognizes the new tool. The gateway provides a specific REST endpoint for clients to discover the capabilities of a given agent model. You can query this endpoint using `curl`.

In your **local terminal**, run this command:

```bash
curl http://localhost:8083/tools/agentic-framework/scientific-agent-v1 | jq
```

In the JSON output, you should now see your new tool listed among the others:
```json
{
  "model_id": "agentic-framework/scientific-agent-v1",
  "tools": [
    ...
    {
      "name": "get_system_information",
      "description": "A tool that retrieves basic system information from the container where the agent gateway is running, such as OS type and architecture.",
      "parameters": {}
    },
    ...
  ]
}
```
This confirms the tool is registered and available to the agent.

### Step 5: Use the New Tool in a Chat Session

Finally, let's use the tool.

1.  Go to the OpenWebUI interface in your browser at `http://localhost:8902`.
2.  Start a new chat.
3.  Ask the agent a question that would cause it to use your new tool. For example:

    > **"What is the system information of the server you are running on?"**

**Expected Agent Behavior:**

1.  The agent will receive your query.
2.  The LLM will analyze the query and see that it matches the `description` of the `get_system_information` tool.
3.  The agent will decide to call this tool.
4.  The `agent_gateway` will execute the command `python3 agentic_framework_pkg/agent_gateway/tools/sys_info_tool.py`.
5.  The script will run inside the `agent_gateway` container, gather the system info, and print it to `stdout` as JSON.
6.  The gateway will return this JSON to the agent.
7.  The agent will receive the JSON and formulate a user-friendly, natural-language response.

The final answer you see in the chat should be something like:

> **"I am running on a Linux operating system with an x86_64 architecture. The Python version is 3.x.x and my container hostname is [some-id]."**

### Progression: From Script to Containerized Tool

The true power of the `stdio` protocol is its flexibility. The gateway doesn't care *what* it runs, only that it can execute the command and communicate via stdin/stdout. We can demonstrate this by packaging our simple Python script into its own standalone Docker container and registering it as a new tool.

This highlights a powerful pattern: you can develop complex tools in any language, package them as container images with all their dependencies, and expose them to the agent without modifying the agent gateway's code.

#### Step 1: Create a Dockerfile for the Tool

A `Dockerfile` is needed to package the `sys_info_tool.py` script. This file has been created for you at `src/agentic_framework_pkg/agent_gateway/tools/Dockerfile.sys_info_tool`.

#### Step 2: Build the Docker Image

On your **remote server**, build the Docker image for the tool. This command must be run from the root of the `agentic_framework` project directory.

```bash
docker build -t sys-info-tool:latest -f docs/tutorial-branches/chapter-06-advanced-multi-agent-orchestration/src/agentic_framework_pkg/agent_gateway/tools/Dockerfile.sys_info_tool docs/tutorial-branches/chapter-06-advanced-multi-agent-orchestration/src/agentic_framework_pkg/agent_gateway/tools/
```

#### Step 3: Create the New JSON Registration File

Now, we create a new registration file for this Dockerized version. The only significant change is the `command` field. This file has been created for you at `src/agentic_framework_pkg/agent_gateway/cli/examples/stdio_sys_info_tool_docker.json`.

Notice the new command:
`"command": ["docker", "run", "--rm", "-i", "sys-info-tool:latest"]`

This tells the gateway to execute the `docker run` command. The `-i` flag is critical, as it connects the gateway's `stdin` to the new container's `stdin`, allowing the JSON input to be passed through.

#### Step 4: Register and Use the Dockerized Tool

Register the new tool using the same `curl` method as before, but pointing to the new JSON file:

```bash
curl -X POST http://localhost:8083/v1/tools/register \
-H "Content-Type: application/json" \
-d @docs/tutorial-branches/chapter-06-advanced-multi-agent-orchestration/src/agentic_framework_pkg/agent_gateway/cli/examples/stdio_sys_info_tool_docker.json
```

After registering, you can verify it by calling the `/tools/{model_id}` endpoint again. Then, in OpenWebUI, you can specifically ask the agent to use this new version:

> **"Use the DOCKERIZED tool to get the system information."**

The agent will now invoke the `docker run` command, and you will get the system information from the ephemeral `sys-info-tool` container, demonstrating the seamless compatibility of the `stdio` protocol.

### Advanced Tutorial: Using the `docker mcp gateway run` Universal Adapter

This final tutorial demonstrates the most advanced and decoupled method for tool invocation: using a universal adapter for the `docker mcp gateway run` command. This pattern provides maximum flexibility, allowing you to add and modify underlying tools without ever changing the registration data in the `agent_gateway`.

**The Concept:**

Instead of registering each tool's specific command (`python3 script.py` or `docker run image`), we will register a single, universal tool called `invoke_mcp_gateway_tool`. This tool's only job is to execute `docker mcp gateway run`, passing along the *actual* tool name and arguments it receives. The `docker mcp` toolkit then becomes responsible for finding and running the target tool.

**Advantages:**
- **Ultimate Decoupling:** The `agent_gateway` no longer needs to know how any specific tool is run. Its only dependency is the `docker mcp` command.
- **Simplified Management:** You can add, remove, or update tools in the environment that `docker mcp` uses without ever needing to re-register them with the `agent_gateway`.

#### Step 1: The Gateway Adapter Script

First, a simple Python script acts as a bridge. The `agent_gateway` will execute this script, which in turn executes the `docker mcp gateway run` command. This provides a clean layer of abstraction. This file has been created for you at `src/agentic_framework_pkg/agent_gateway/tools/gateway_run_adapter.py`.

#### Step 2: The Universal Tool Registration File

Next, we create the JSON configuration to register our universal adapter. This is the key part that exposes the `docker mcp gateway run` functionality to the agent. This file has been created for you at `src/agentic_framework_pkg/agent_gateway/cli/examples/stdio_docker_mcp_gateway_run_example.json`.

**Key `args_schema` fields:**
- `mcp_tool_name`: The agent must provide the name of the *actual* tool it wants to run (e.g., `get_system_information`).
- `mcp_tool_args`: The agent must provide a JSON object containing the arguments for that target tool.

#### Step 3: Register the Universal Adapter

Using your local, port-forwarded terminal, register this new universal tool.

```bash
curl -X POST http://localhost:8083/v1/tools/register \
-H "Content-Type: application/json" \
-d @docs/tutorial-branches/chapter-06-advanced-multi-agent-orchestration/src/agentic_framework_pkg/agent_gateway/cli/examples/stdio_docker_mcp_gateway_run_example.json
```

#### Step 4: Use the Universal Adapter in a Chat Session

Now, you can invoke *any* tool known to the `docker mcp` environment through this single, unified interface.

Go to OpenWebUI and ask the agent:

> **"Use the `invoke_mcp_gateway_tool` to run the `get_system_information` tool with no arguments."**

**Expected Agent Behavior:**

1.  The agent will call the `invoke_mcp_gateway_tool`.
2.  It will provide the parameters: `mcp_tool_name="get_system_information"` and `mcp_tool_args={}`.
3.  The `agent_gateway` runs the `gateway_run_adapter.py` script.
4.  The adapter script executes `docker mcp gateway run get_system_information`.
5.  The `docker mcp` toolkit finds and runs the `get_system_information` tool.
6.  The result is passed all the way back up the chain to the agent, which then formulates the final answer.

This demonstrates a highly flexible and maintainable architecture for managing and exposing tools to your AI agents.

### Tutorial: Registering and Using an HTTP-Based Tool

This tutorial showcases how to register a tool that communicates with an external service over HTTP, allowing the agent to interact with existing microservices or public APIs.

**Our Goal:** We will register a tool called `get_weather_conditions`. When used, the Agent Gateway will make an HTTP `GET` request to a hypothetical external weather API.

#### Step 1: Define the Target Service

For this tutorial, we will imagine there is an external weather API at `https://api.example.com/weather/current` that requires an `X-API-Key` header and accepts a `city` parameter.

#### Step 2: Create the HTTP Tool Registration File

Next, we create a JSON file that tells the gateway how to call this HTTP endpoint. This file, `http_tool_example.json`, already exists in the project and serves as a template.

**Key `invocation` Fields:**
- **`protocol`**: Must be `"http"`.
- **`connection.url`**: The full URL of the API endpoint.
- **`connection.method`**: The HTTP method to use (e.g., `GET`, `POST`).
- **`connection.headers`**: A dictionary of headers for things like authentication.

#### Step 3: Register the HTTP Tool

In your **local terminal** (with the SSH port forward active), run the `curl` command to register the tool:

```bash
curl -X POST http://localhost:8083/v1/tools/register \
-H "Content-Type: application/json" \
-d @docs/tutorial-branches/chapter-06-advanced-multi-agent-orchestration/src/agentic_framework_pkg/agent_gateway/cli/examples/http_tool_example.json
```

#### Step 4: Verify and Use the Tool

1.  **Verify**: Use `curl http://localhost:8083/tools/agentic-framework/scientific-agent-v1 | jq` to confirm the `get_weather_conditions` tool is listed.
2.  **Use**: Go to the OpenWebUI and ask the agent, **"What is the weather like in Seattle?"**

**Expected Behavior:**
The agent will call the tool. The gateway will then make a `GET` request to the configured URL. While this specific example will fail (as the API is hypothetical), it demonstrates the mechanism for integrating real-world APIs.


```

## Demonstration Scenarios

This section provides sample queries to demonstrate the advanced multi-agent orchestration capabilities of this chapter, including how they interact with other tools and handle plotting.

### Scenario 1: Router Mode - Collaborative "Plan and Review"

This scenario demonstrates the new, more robust "Plan and Review" workflow in 'router' mode.

1.  **Create Multi-Agent Session:**
    *   **User Query:** "Create a multi-agent session in 'router' mode. The team should include a 'Bioinformatician', a 'Chemist', and a 'Software Engineer'. The overall goal is to perform a competitive analysis for a new drug candidate targeting protein P01112."
    *   **Expected Agent Response:** The agent will confirm session creation and provide a `multi_agent_session_id`.

2.  **Generate and Review Plan:**
    *   **User Query:** "Using the 'Plan and Review' process, generate a detailed, multi-phase plan for the session I just created, focusing on analyzing P01112's sequence, finding existing patents, and identifying similar compounds. Ensure the plan uses appropriate MCP tools and includes Python code execution steps for plotting where relevant. For Python plotting, remember to explicitly create figure objects (e.g., `fig, ax = plt.subplots()`) and avoid `plt.show()`. The figure object should be the last expression in the code block for capture. Also, ensure any Python code for plotting includes in-script package installations (e.g., using `subprocess`)."
    *   **Expected Agent Response:** The agent will use the `generate_and_review_plan_for_multi_agent_task` tool. It will then present the refined, team-vetted plan and ask for your approval.

3.  **Approve and Execute Plan:**
    *   **User Query:** "The plan looks good. Approve and execute it."
    *   **Expected Agent Behavior:** The agent will execute the plan. If a plotting step is encountered, the `ExecuteCode` tool runs the Python code, captures the plot, and the agent includes the plot URL in its response, which should then render in the UI.

### Scenario 2: Graph Mode - Dynamic Problem Solving with Loop Prevention

This scenario demonstrates the dynamic 'graph' mode, now with enhanced resilience against infinite loops.

1.  **Create Multi-Agent Session:**
    *   **User Query:** "Create a multi-agent session in 'graph' mode. The team should include a 'Biologist' and a 'Chemist'."
    *   **Expected Agent Response:** The agent will confirm session creation and provide a `multi_agent_session_id`.

2.  **Execute Dynamic Task:**
    *   **User Query:** "Using the graph-based session, find the human KRAS protein sequence from UniProt, then perform a BLAST search against SwissProt, and finally identify chemical compounds in PubChem known to inhibit it. Include a plot of the BLAST results if possible, remembering the plotting guidelines for Python code."
    *   **Expected Agent Behavior:** The supervisor agent dynamically routes tasks to the 'Biologist' (for UniProt and BLAST) and 'Chemist' (for PubChem). The supervisor will monitor the `agent_turn_count` to prevent any single agent from getting stuck in a loop. The final response should synthesize findings and include any generated plot, which should then render correctly.

### Scenario 3: n8n Workflow Integration

This scenario demonstrates how to use the `ScientificWorkflowAgent` within an n8n workflow to automate a research task.

1.  **Start the Services:**
    *   Run `./start-chapter-resources.sh`. The script will automatically start the n8n container alongside all other services.
    *   > **Troubleshooting:** If you see an `authentication required` error for the `n8nio/n8n` image, it means your local Docker environment has stale credentials. The quickest fix is to log out of Docker Hub by running `docker logout` in your terminal. This will allow Docker to pull the public image anonymously. If the issue persists (due to rate-limiting), run `docker login` and enter your Docker Hub credentials.

2.  **Access n8n:**
    *   Open your web browser and navigate to `http://localhost:5678`.

3.  **Configure the n8n `AI Agent` Node:**
    *   Create a new workflow in n8n.
    *   Add the **`AI Agent`** node.
    *   In the "LLM" section, select "Chat Model" and choose to **Create New Credentials**.
    *   Select "OpenAI API" for the credential type.
    *   **Base URL:** Set this to `http://agent_gateway:8081/v1`. This uses Docker's internal networking to connect to the agent's backend.
    *   **API Key:** Enter a dummy value (e.g., "none").
    *   **Model Name:** Enter the model ID exposed by the backend: `agentic-framework-cot`.

4.  **Build a Workflow:**
    *   Create a workflow that starts with a "Manual" trigger.
    *   Connect the trigger to the `AI Agent` node.
    *   In the `AI Agent` node's "Prompt" field, enter a query for the agent. For example: *"Find the UniProt accession ID for human BRCA1, then get its full sequence."*

5.  **Execute and Observe:**
    *   Run the workflow. The `AI Agent` node will send the prompt to your `ScientificWorkflowAgent`. The agent will use its tools (`QueryUniProt`) to find the answer and return the result as the output of the n8n node. You can then connect this output to other n8n nodes (e.g., save to a Google Sheet, send an email).

---

**Important Note for Plot Rendering:**

For plot rendering to work correctly in your local Docker Compose environment, ensure you have set the `SANDBOX_MCP_SERVER_PUBLIC_URL` environment variable in your `.env` file (e.g., `SANDBOX_MCP_SERVER_PUBLIC_URL=http://localhost:8082`). This variable tells the sandbox server the public URL it should use when generating plot links.

## References

- [n8n Documentation](https://docs.n8n.io/getting-started/introduction/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)

## Additional Resources

### Vector Databases and RAG

- **Coursera**: [Building and Deploying Vector Search Applications](https://www.coursera.org/learn/classification-vector-spaces-in-nlp)
- **Udemy**: [Vector Databases: From Zero to Hero](https://www.udemy.com/course/vector-databases-ai/)
- **YouTube**: [What are Vector Embeddings?](https://www.youtube.com/watch?v=NEreO2zlXDk)
- **YouTube**: [Retrieval Augmented Generation (RAG) Explained](https://www.youtube.com/watch?v=T-D1OfcDW1M)

### Multi-Agent Systems

- **YouTube**: [Multi-Agent Reinforcement Learning](https://www.youtube.com/watch?v=QfYx5q0Q75M)
- **Udemy**: [Multi-Agent Systems with Python](https://www.udemy.com/course/multi-agent-systems-with-python/)
- **YouTube**: [Introduction to Multi-Agent Systems](https://www.youtube.com/watch?v=eHEHE2fpnWQ)
