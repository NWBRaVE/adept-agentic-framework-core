# ADEPT: Agentic Discovery and Exploration Platform for Tools Tutorial Outline

This document outlines the chapters of the Agentic Framework tutorial. Each chapter builds upon the last, introducing new features and concepts.

---

## Chapter 00: Introduction

This chapter introduces the Agentic Discovery and Exploration Platform for Tools (ADEPT), a teaching tool and blueprint for integrating Large Language Models (LLMs) with scientific computing workflows. The framework is designed as a hands-on guide to foster a shared understanding of agentic AI systems. It provides a modular, multi-service architecture that runs locally, demonstrating how to wrap familiar scientific tools for use by an AI agent.

### Key Concepts

- **Agentic Framework**: A modular framework for building agentic applications.
- **Model Context Protocol (MCP)**: A standard for hosting tools and managing state, implemented here with `fastmcp`.
- **Docker**: Used to containerize each service for a consistent, reproducible environment.
- **Langchain**: The core framework for building the agent's reasoning and tool-use logic.
- **Streamlit & JupyterLab**: Dual user interfaces for both conversational interaction (Streamlit) and direct, code-based access (JupyterLab).
- **RAG (Retrieval-Augmented Generation)**: The underlying technique for providing the agent with knowledge from uploaded files and notes.

### Core Components

- **User Interfaces**: Streamlit App for chat and JupyterLab for direct, code-based interaction.
- **Agent Orchestration (Langchain Agent)**: The central "brain" of the application, using LangChain for reasoning and tool use.
- **Tool Execution Layer (MCP Server)**: A FastAPI-based server hosting tools according to the Model Context Protocol (MCP).
- **LLM Provider (Ollama Server)**: Runs open-source LLMs locally to power the agent.
- **Data Persistence**:
    - **SQLite**: For structured, relational data ingested from files.
    - **ChromaDB**: A vector store for the agent's long-term memory (e.g., chat history, notes).

### Example MCP Tools

- **`ingest_data`**: Parses and loads data from CSV or Excel files into a SQLite database.
- **`get_sql_schema`**: Retrieves the database schema so the agent can understand the table structure.
- **`execute_sql`**: Executes a SQL query against the database.
- **`save_note`**: Saves a piece of text to the agent's long-term memory (ChromaDB).
- **`list_notes`**: Retrieves all notes from the agent's memory.

### Getting Started

To run this chapter, navigate to the `chapter-00-introduction` directory and execute the lifecycle script:

```bash
./start-chapter-resources.sh
```

---

## Chapter 01: Main - Basic Architecture

This chapter presents the foundational architecture of the ADEPT (Agentic Discovery and Exploration Platform for Tools). At this stage, the framework is simple and demonstrates the core concepts of an agentic system.

### Architecture

- **Langchain Agent**: A basic [Langchain](https://www.langchain.com/) agent is used for orchestration. It can understand user queries and use tools to fulfill requests.
- **No Chain-of-Thought (CoT)**: The agent in this version is a simple ReAct agent, which is a form of CoT, but for the purpose of this tutorial, we consider it a baseline before introducing more complex reasoning patterns with LangGraph.
- **MCP Session ID**: The `mcp_session_id` is introduced to manage state and context across tool calls.
- **Core Components**: The main components are the [Streamlit](https://streamlit.io/) UI, the Langchain agent, and a single MCP server hosting a basic set of tools.

### Getting Started

To run this chapter, navigate to the `chapter-01-main` directory and execute the lifecycle script:

```bash
./start-chapter-resources.sh
```

---

## Chapter 02: HPC MCP Server and Chain-of-Thought

This chapter enhances the ADEPT framework by introducing a dedicated MCP server for [High-Performance Computing (HPC)](https://www.hpc.llnl.gov/documentation/tutorials) tasks and upgrading the agent to use [LangGraph](https://langchain-ai.github.io/langgraph/) for more sophisticated [Chain-of-Thought (CoT)](https://www.promptingguide.ai/techniques/cot) reasoning.

### Key Additions

- **HPC MCP Server**: A new, separate `fastmcp` server is added to host computationally intensive tools like [Nextflow](https://www.nextflow.io/) pipelines for [BLAST](https://blast.ncbi.nlm.nih.gov/Blast.cgi) searches, video processing with [Whisper](https://github.com/openai/whisper), and code repository security scanning with [GitXRay](https://www.gitxray.com/). This separation of concerns prevents long-running tasks from blocking the main MCP server. The server is built with [FastAPI](https://fastapi.tiangolo.com/) and run with [Uvicorn](https://www.uvicorn.org/).
- **LangGraph Integration**: The agent is upgraded from a simple agent to a more complex reasoning engine built with [LangGraph](https://langchain-ai.github.io/langgraph/) and [LangChain](https://www.langchain.com/). This allows for more explicit and controllable multi-step reasoning, which is a more advanced form of [Chain-of-Thought](https://www.promptingguide.ai/techniques/cot). The agent can now create more complex plans and execute them.
- **Stateful Interactions**: The use of `mcp_session_id` is continued and becomes more important for tracking the state of these more complex, multi-step workflows. We use [ChromaDB](https://docs.trychroma.com/) for session state management.

### Getting Started

To run this chapter, navigate to the `chapter-02-hpc-mcp-server-with-cot` directory and execute the lifecycle script:

```bash
./start-chapter-resources.sh
```

---

## Chapter 03: LLM Sandbox and Multi-Agent Capabilities

This chapter introduces two major features to the ADEPT framework: a sandboxed environment for secure code execution and a multi-agent system for tackling complex tasks.

### Key Additions

- **Sandbox MCP Server**: A new, highly specialized MCP server is added to execute arbitrary Python, JavaScript, and shell code in a secure, isolated environment using the `llm-sandbox` library. This allows the agent to perform dynamic calculations, data manipulation, and other tasks that can be solved with code, without compromising the host system.
- **Multi-Agent Tool**: The main MCP server is updated with a `multi_agent_tool`. This tool allows for the creation and management of teams of AI agents. A "Planner" agent can generate a plan, which is then executed by a "Supervisor" agent that delegates tasks to worker agents with specific roles (e.g., "chemist", "bioinformatician"). This enables the framework to handle much more complex, multi-step workflows that require different areas of expertise.
- **Multi-Agent ID**: A `multi_agent_id` is introduced for context tracking within these complex, multi-agent sessions.

### Getting Started

To run this chapter, navigate to the `chapter-03-llm-sandbox-and-multi-agent` directory and execute the lifecycle script:

```bash
./start-chapter-resources.sh
```

---

## Chapter 04: Cloud-Native Deployment

This chapter transitions the ADEPT framework from a local Docker Compose setup to a cloud-native deployment model, targeting Kubernetes. It introduces the necessary Infrastructure as Code (IaC) and CI/CD components to automate the deployment and management of the entire application stack on both Microsoft Azure and Amazon Web Services (AWS).

### Key Additions

- **Kubernetes Deployment with Helm**: A comprehensive Helm chart is introduced in `infra/helm/` to define, configure, and deploy all the framework's services (MCP servers, Streamlit UI, etc.) as Kubernetes resources. This enables consistent and repeatable deployments across different Kubernetes environments.

- **Azure Infrastructure & CI/CD**:
    - **Pulumi for Azure**: The `infra/azure/pulumi/` directory contains a Pulumi project to programmatically provision the core Azure infrastructure, including an Azure Kubernetes Service (AKS) cluster and an Azure Container Registry (ACR).
    - **Azure Pipelines**: An `azure-pipelines.yml` file defines a full CI/CD pipeline. This pipeline automates the process of building all service Docker images, pushing them to ACR, and deploying the Helm chart to the AKS cluster.

- **AWS Infrastructure & CI/CD**:
    - **AWS CDK**: The `infra/aws/cdk/` directory provides an AWS Cloud Development Kit (CDK) application to provision the necessary AWS infrastructure. This includes Amazon ECR repositories for container images and an Amazon ECS cluster for running the services.
    - **AWS CodePipeline**: The CDK app also sets up a complete CI/CD pipeline using AWS CodePipeline and AWS CodeBuild, which automates the building, pushing, and deployment of the services to ECS.

### Getting Started

To run this chapter, navigate to the `chapter-04-kubernetes-deployment` directory and execute the lifecycle script:

```bash
./start-chapter-resources.sh
```

---

## Chapter 05: OpenWebUI Integration

This chapter integrates the ADEPT framework with OpenWebUI, providing a polished, production-grade user interface for interacting with the scientific workflow agent. It also introduces a new backend service to make this integration possible.

### Key Additions

- **OpenWebUI Backend Service**: A new service, `openwebui_backend`, is introduced. It is defined in `Dockerfile.openwebui_backend` and orchestrated via `docker-compose-openwebui.yaml`. This service exposes an OpenAI-compatible API endpoint (`/v1/chat/completions`), allowing the `ScientificWorkflowAgent` to act as a backend model for any OpenAI-compatible frontend.

- **Agent as a Backend Model**: The new backend service enables the agent to be seamlessly integrated into OpenWebUI. Users can select the "Agentic Framework CoT" model from the dropdown in OpenWebUI to interact directly with the full suite of scientific and computational tools.

- **Frontend/Backend Separation**: This chapter emphasizes a clear architectural separation between the user interface and the agentic backend. The `docker-compose-openwebui.yaml` file manages the backend services, while the OpenWebUI container runs as a separate frontend, communicating with the backend via the standardized API. This modularity allows for independent development, scaling, and maintenance of the UI and the agent framework.

### Getting Started

To run this chapter, navigate to the `chapter-05-openwebui-integration` directory and execute the lifecycle script:

```bash
./start-chapter-resources.sh
```

---

## Chapter 06: The Agent Gateway - Dynamic Tools and Universal Integration

This chapter culminates the tutorial by introducing the **Agent Gateway**, a sophisticated service that transforms the framework into a robust, extensible, and production-ready system. It provides a unified, secure, and protocol-agnostic entry point for all agent interactions, enabling dynamic tool registration and seamless integration with multiple frontends and workflow engines.

### Key Architectural Patterns

- **Agent as a Universal Backend**: The gateway exposes a single, OpenAI-compatible API. This allows diverse clients like **OpenWebUI** (for chat) and **n8n** (for workflow automation) to use the `ScientificWorkflowAgent` as a powerful, tool-enabled "brain" without any custom integration code.
- **Dynamic Tool Provisioning via `stdio`**: The gateway can dynamically register new tools at runtime without a restart. It uses a flexible `stdio` protocol, allowing it to execute any command-line tool (from a Python script to a Docker container) that can communicate over standard input/output. This provides a powerful mechanism for extending the agent's capabilities on the fly.
- **Enhanced Multi-Agent Orchestration**: The multi-agent system is upgraded with two distinct orchestration modes:
    - **Router Mode with "Plan and Review"**: A deterministic, plan-based approach where an agent team collaboratively reviews and refines a plan before it is presented to the user for approval.
    - **Graph Mode with Loop Prevention**: A dynamic, `langgraph`-based supervisor that can handle non-linear, complex tasks by routing between agents and preventing infinite loops.
- **Centralized Vector Database**: A standalone ChromaDB server provides a persistent and scalable vector store for all RAG and embedding-based tools.

### Hands-On Tutorials

This chapter includes a series of hands-on tutorials that demonstrate the gateway's core functionalities:

1.  **Dynamically Registering a `stdio` Tool**: Learn how to write a simple Python script and a corresponding JSON configuration to register it as a new tool that the agent can use immediately.
2.  **Progressing to a Containerized Tool**: Extend the first tutorial by packaging the same Python script into a standalone Docker container. You will learn how to update the tool's registration to have the gateway execute your tool via a `docker run` command, demonstrating the power and flexibility of the `stdio` protocol.
3.  **Using the Universal Gateway Adapter**: Explore the most advanced invocation pattern by registering a universal `invoke_mcp_gateway_tool`. This tool uses the `docker mcp gateway run` command to act as a proxy, allowing the agent to run *any* tool available to the MCP toolkit without requiring individual registration for each one.

### Getting Started

To run this chapter, navigate to the `chapter-06-advanced-multi-agent-orchestration` directory and execute the lifecycle script:

```bash
./start-chapter-resources.sh
```
