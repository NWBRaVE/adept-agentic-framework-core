
# Proposal: Integrating the MCP Agentic Framework with n8n

## 1. Executive Summary

This document outlines a clear, multi-tiered strategy for integrating the Multi-Agent Communication Protocol (MCP) agentic framework with the n8n workflow automation platform. The integration leverages n8n's advanced AI capabilities, including its native support for LangChain.

The cornerstone of all integration strategies is the **`openwebui_backend` service**. This service exposes an OpenAI-compatible API that allows the powerful, tool-rich `ScientificWorkflowAgent` to be seamlessly plugged into n8n.

### Integration Strategies:
1.  **Direct API Connection (Good for Simple Tasks):** The fastest approach. Connect n8n's generic "OpenAI" node to your agent's API. Ideal for simple, single-turn queries.
2.  **Native LangChain Agent Integration (Recommended for Advanced Workflows):** The most powerful approach. Use n8n's built-in `AI Agent` node, configuring it to use your `ScientificWorkflowAgent` as its core "brain." This unlocks the full potential of n8n's AI features (memory, RAG, etc.) while leveraging your entire MCP tool ecosystem.
3.  **Custom n8n Node (Ultimate UX):** The most tailored approach. Develop a dedicated n8n node for the ultimate user experience, but with significant development overhead.

**Recommendation:** Start with Strategy 1 for immediate results and testing. For all serious workflow development, **adopt Strategy 2**, as it provides the best balance of power, flexibility, and deep integration with the n8n ecosystem.

## 2. Analysis of n8n & The Integration Opportunity

n8n is a "fair-code" licensed tool for building complex workflows via a visual interface. My analysis, confirmed by n8n's official documentation, reveals robust support for:

1.  **Connecting to External APIs:** n8n has pre-built nodes for interacting with LLMs like OpenAI.
2.  **Native LangChain Support:** n8n provides first-class "AI Agent" and "Chain" nodes built on LangChain. This allows for sophisticated agentic workflows, including Retrieval-Augmented Generation (RAG) using various vector stores, all within the n8n environment.
3.  **Custom Node Development:** A well-documented SDK (`n8n-nodes-starter`) exists for creating bespoke nodes with TypeScript.

Your framework is perfectly positioned for this. The `openwebui_backend` service already emulates the OpenAI `/v1/chat/completions` endpoint, making it the key to unlocking all three integration strategies.

## 3. Strategy 1: Direct API Connection (Simple)

This is the most straightforward method to get started. It treats your `ScientificWorkflowAgent` as a standard LLM that n8n can call.

### How It Works

The `openwebui_backend` (`app.py`) serves an API that is a drop-in replacement for the OpenAI API. We configure n8n's existing LLM node to point to your local `openwebui_backend` service instead of the public OpenAI servers.

### Architectural Diagram

```
+-----------------+      +----------------------+      +---------------------+      +---------------------------+
|     n8n UI      |----->| n8n Workflow Engine  |----->|  openwebui_backend  |----->|  ScientificWorkflowAgent  |
| (User Workflow) |      | (Generic OpenAI Node)|      | (OpenAI-compatible) |      | (LangGraph & MCP Tools)   |
+-----------------+      +----------------------+      +---------------------+      +---------------------------+
```

### Step-by-Step Implementation Guide

1.  **Start the Agent Framework Services:**
    Navigate to the `tutorial-branches/chapter-06-advanced-multi-agent-orchestration/` directory and launch all required services using the new `docker-compose-n8n.yaml` file.

    ```bash
    # Ensure all services, including the backend and n8n, are running
    docker compose -f docker-compose.yaml -f docker-compose-openwebui.yaml -f docker-compose-hpc.yaml -f docker-compose-n8n.yaml up --build -d
    ```
    > **Troubleshooting:** If you see an `authentication required` error for the `n8nio/n8n` image, it means your local Docker environment has stale credentials. The quickest fix is to log out of Docker Hub by running `docker logout` in your terminal. This will allow Docker to pull the public image anonymously. If the issue persists (due to rate-limiting), run `docker login` and enter your Docker Hub credentials.

2.  **Configure the n8n LLM Node:**
    *   Open your n8n instance (e.g., `http://localhost:5678`).
    *   Create a new workflow.
    *   Add a node. Search for "OpenAI", "Chat Model", or a similar generic LLM node.
    *   **Create a New Credential:**
        *   In the node's credential settings, select "Create New".
        *   Choose an option like "OpenAI API".
        *   **Base URL:** This is the most critical step. Because n8n is now on the same Docker network as the agent backend, you can use the service name directly. Set the URL to `http://openwebui_app:8081/v1`. (Adjust port if necessary).
        *   **API Key:** Your `openwebui_backend` does not require an API key. You can enter any dummy value (e.g., "none").
    *   **Configure the Node Parameters:**
        *   **Model:** In the model field, enter the exact model ID exposed by your backend: `agentic-framework-cot`.
        *   **Input:** Map the input from a previous node or a trigger to the "Prompt" field.

3.  **Run the Workflow:**
    Execute the workflow. The n8n node will make a request to your `openwebui_backend`, which will invoke the `ScientificWorkflowAgent`. The agent's final response will be returned as the output of the n8n node.

## 4. Strategy 2: Native LangChain Agent Integration (Recommended)

This strategy leverages n8n's native `AI Agent` node to create more powerful and stateful workflows. It treats your `ScientificWorkflowAgent` as a custom, tool-aware "brain" that drives the n8n agent.

### How It Works

n8n's `AI Agent` node is a pre-built LangChain agent executor. It can be configured with an LLM, tools, and memory. We will configure it to use your `openwebui_backend` as its LLM. In this model, n8n manages the agent loop, while your service provides the reasoning and access to the entire MCP tool ecosystem.

### Architectural Diagram

```
+-----------------+      +----------------------+      +---------------------+      +---------------------------+
|     n8n UI      |----->|   n8n AI Agent Node  |----->|  openwebui_backend  |----->|  ScientificWorkflowAgent  |
| (User Workflow) |      | (Manages Agent Loop) |      | (Acts as Custom LLM)|      | (Executes Tools via MCP)  |
+-----------------+      +----------------------+      +---------------------+      +---------------------------+
```

### Step-by-Step Implementation Guide

1.  **Start Services:** Follow step 1 from Strategy 1 to launch the entire stack.

2.  **Configure the n8n `AI Agent` Node:**
    *   In a new n8n workflow, add the **`AI Agent`** node.
    *   **LLM:** In the "LLM" section of the node, choose the "Chat Model" option.
        *   **Create a New Credential:** Select the "OpenAI API" option.
        *   **Base URL:** Set this to your backend service using the Docker service name: `http://openwebui_app:8081/v1`.
        *   **API Key:** Enter a dummy value (e.g., "none").
        *   **Model Name:** Enter the model ID from your backend: `agentic-framework-cot`.
    *   **Tools:** For this strategy, you do **not** need to add tools directly to the n8n node. Your `ScientificWorkflowAgent` already has its own tools (the MCP toolset) and will report its actions back to n8n as part of its response.
    *   **Input:** Connect the user's query to the `AI Agent` node's "Input".

### Benefits Over Direct API Connection
*   **State Management:** Easily leverage n8n's built-in memory options to have multi-turn conversations.
*   **RAG Integration:** Seamlessly connect the agent to n8n's vector store nodes (`Chroma`, `Pinecone`, etc.) to build powerful RAG applications.
*   **Ecosystem Compatibility:** Better integration with the broader n8n ecosystem, allowing your agent to drive workflows that use other n8n nodes.

## 5. Strategy 3: Custom n8n Node (Advanced)

For a more deeply integrated and user-friendly experience, you can develop a dedicated n8n node for the "MCP Agentic Framework". This remains the "version 2.0" path.

### Benefits:

*   **Tailored UI:** Create specific input fields for multi-agent parameters like `roles` or `execution_mode` instead of having the user type them in a generic prompt.
*   **Simplified Configuration:** Hide the complexity of the API endpoint and model selection from the end-user.
*   **Custom Output Handling:** Natively handle complex outputs from your agent, such as structured data or plot URLs, and format them correctly within n8n.

### Development Path:

1.  **Use the Starter Template:** Clone the `n8n-nodes-starter` repository.
2.  **Define the Node:** Create a `Node.ts` file that defines the UI properties.
3.  **Implement the `execute` method:** This method will use n8n's built-in HTTP request functionality to call your `openwebui_backend` API.
4.  **Package and Deploy:** Follow the n8n documentation to build, test, and link your custom node to your n8n instance.

## 6. Conclusion

By leveraging the existing OpenAI-compatible API provided by the `openwebui_backend`, you can integrate your sophisticated multi-agent framework into n8n's visual workflow builder **immediately**. The native `AI Agent` node in n8n provides the ideal entry point for this integration.

This powerful combination allows you to build complex, automated workflows that are orchestrated by your intelligent `ScientificWorkflowAgent`, blending the best of both platforms.
