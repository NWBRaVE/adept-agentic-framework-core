# Chapter 01: Main - Basic Architecture

## Getting Started

To run this chapter, navigate to the current directory and execute the lifecycle script. This script will handle all the Docker services for you.

```bash
./start-chapter-resources.sh
```

---

This chapter presents the foundational architecture of ADEPT (Agentic Discovery and Exploration Platform for Tools). At this stage, the framework is simple and demonstrates the core concepts of an agentic system.

## Architecture

- **Langchain Agent**: A basic [Langchain](https://www.langchain.com/) agent is used for orchestration. It can understand user queries and use tools to fulfill requests.
- **No Chain-of-Thought (CoT)**: The agent in this version is a simple ReAct agent, which is a form of CoT, but for the purpose of this tutorial, we consider it a baseline before introducing more complex reasoning patterns with LangGraph.
- **MCP Session ID**: The `mcp_session_id` is introduced to manage state and context across tool calls.
- **Core Components**: The main components are the [Streamlit](https://streamlit.io/) UI, the Langchain agent, and a single MCP server hosting a basic set of tools.

This chapter serves as the starting point for the tutorial, upon which more advanced features will be added in subsequent chapters.

### How It Works: The Main MCP Server

This chapter introduces the main `mcp_server`, which hosts the core scientific and general-purpose tools that form the foundation of the agent's capabilities.

**Example User Query:** "I have uploaded a file containing protein data. What can you tell me about the protein with accession number P05067?"

1.  **File Upload**: The user first uploads a file (e.g., a CSV or PDF) containing protein data using the Streamlit UI. The `process_uploaded_file` tool is called, which processes the document and returns a `file_id`.

2.  **Tool Selection**: The agent receives the user's query and determines that it needs to consult the uploaded file. It first calls the `ListUploadedFiles` tool to see what files are available for the current session and to confirm the correct `file_id`.

3.  **RAG Query**: The agent then calls the `QueryProcessedDocumentData` tool, providing the `file_id` and the user's query ("protein with accession number P05067").

4.  **Data Retrieval and Synthesis**: The tool performs a similarity search within the specified file's indexed content, retrieves the relevant information about protein P05067, and passes it back to the agent. The agent then synthesizes this information into a final answer for the user.

## Key Technologies Used

This project leverages several key technologies to build the agentic framework:

-   **[LangChain](https://www.langchain.com/)**: A framework for developing applications powered by language models.
-   **[Streamlit](https://streamlit.io/)**: An open-source app framework for Machine Learning and Data Science teams.
-   **[FastMCP](https://gofastmcp.com/)**: A Python framework for building Model Context Protocol (MCP) servers and clients, enabling LLMs to interact with external tools.
-   **[FastAPI](https://fastapi.tiangolo.com/)**: A modern, fast (high-performance), web framework for building APIs with Python 3.7+ based on standard Python type hints.
-   **[Docker](https://docs.docker.com/)**: A platform for developing, shipping, and running applications in containers.
-   **[LiteLLM](https://docs.litellm.ai/)**: A library that provides a unified interface to over 100 LLM APIs.
-   **[ChromaDB](https://docs.trychroma.com/)**: An open-source embedding database for building AI applications with semantic search.
-   **[Biopython](https://biopython.org/)**: A set of freely available tools for biological computation written in Python.
-   **[PubChemPy](https://pubchempy.readthedocs.io/)**: A Python wrapper for the PubChem PUG REST API.
-   **[UniProt](https://www.uniprot.org/)**: A comprehensive resource for protein sequence and annotation data.
-   **[Selenium](https://selenium-python.readthedocs.io/)**: A powerful tool for controlling web browsers through programs and performing browser automation.
-   **[NVIDIA NIM](https://developer.nvidia.com/nim)**: NVIDIA Inference Microservices (NIM) provide optimized, easy-to-use microservices for deploying generative AI models.
-   **[Uvicorn](https://www.uvicorn.org/)**: An ASGI web server implementation for Python.

## Additional Resources

### Key Concepts

-   **Agentic Systems**:
    -   [What is Agentic AI? (AWS)](https://aws.amazon.com/what-is/agentic-ai/)
    -   [Building Agentic AI Systems from Scratch (Medium)](https://medium.com/@bhavik.maru/how-to-build-agentic-ai-systems-from-scratch-a-practical-guide-for-developers-5a7e54a5c32e)
-   **Retrieval-Augmented Generation (RAG)**:
    -   [Retrieval-Augmented Generation (RAG): A Comprehensive Guide (Datacamp)](https://www.datacamp.com/courses/retrieval-augmented-generation-rag-with-langchain)
    -   [Building RAG Applications with LangChain (YouTube)](https://www.youtube.com/watch?v=tcqEUSNCn8I)
-   **Chain-of-Thought (CoT) Prompting**:
    -   [Chain-of-Thought Prompting (PromptHub)](https://prompthub.us/chain-of-thought-prompting-a-complete-guide-for-beginners/)
    -   [Chain of Thought Prompting Explained (YouTube)](https://www.youtube.com/watch?v=Kar2qfLDQ2c)

### Technology Tutorials

-   **LangChain**:
    -   [LangChain for Beginners Crash Course (YouTube)](https://www.youtube.com/watch?v=L_Guz73e6fw)
    -   [Official LangChain Tutorials](https://python.langchain.com/docs/get_started/introduction)
-   **Streamlit**:
    -   [Streamlit for Beginners (YouTube)](https://www.youtube.com/watch?v=D0D4Pa22iG0&t=41s)
    -   [Official Streamlit Tutorials](https://docs.streamlit.io/library/get-started)
-   **FastAPI**:
    -   [FastAPI - The Complete Course 2025 (Udemy)](https://www.udemy.com/course/fastapi-the-complete-course/?kw=fast&src=sac)
    -   [Official FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/)
-   **Docker**:
    -   [Docker for Beginners (Coursera)](https://www.coursera.org/learn/docker-for-the-absolute-beginner)
    -   [Docker Tutorial for Beginners (YouTube)](https://www.youtube.com/watch?v=3c-iBn73dDE)
-   **ChromaDB**:
    -   [ChromaDB Crash Course (YouTube)](https://www.youtube.com/watch?v=ySus5ZS0b94)
    -   [Official ChromaDB Examples](https://docs.trychroma.com/examples)
