
# Agentic Framework Tool User Guide

This guide provides a detailed explanation of how the various tools within the Agentic Framework operate. Each section corresponds to a specific MCP (Modular Command Processor) tool server and its associated tools, outlining the data flow from user query to final response.

## 1. Main MCP Server (`mcp_server`)

The main MCP server hosts the core scientific and general-purpose tools that form the foundation of the agent's capabilities.

### 1.1. File Processing & RAG

These tools enable the agent to ingest documents, create a searchable knowledge base, and answer questions based on their content.

**Tools:**
- `process_uploaded_file`
- `query_file_content`
- `list_uploaded_files`

#### How Does It Work?

**Example User Query:** "I've uploaded a research paper in PDF format. Can you tell me what it says about protein kinase inhibitors?"

1.  **File Upload and Processing:**
    *   The user uploads a file (e.g., a PDF, CSV, or DOCX) through the user interface.
    *   The UI saves the file to a shared volume, making it accessible to the MCP server.
    *   The `process_uploaded_file` tool is invoked with the file's path and original name.

2.  **Content Extraction and Chunking:**
    *   The tool reads the document and extracts its text content. For complex formats like PDFs, it uses libraries like PyMuPDF to ensure accurate text extraction.
    *   The extracted text is then split into smaller, overlapping chunks to prepare it for embedding.

3.  **Vector Embedding and Indexing:**
    *   Each text chunk is converted into a numerical representation (a vector embedding) using a specialized embedding model.
    *   The `VectorStoreManager` creates a unique, isolated collection in the ChromaDB vector store for this specific file. The collection's name is a combination of the user's session ID and a newly generated `file_id`.
    *   The text chunks and their corresponding embeddings are stored in this collection.

4.  **Agent-Driven Querying:**
    *   When the user asks a question about the document, the `ScientificWorkflowAgent` identifies the intent to query the file and selects the `query_file_content` tool.
    *   The agent passes the user's query and the relevant `file_id` to the tool.

5.  **Similarity Search and RAG:**
    *   The `query_file_content` tool uses the `file_id` to identify the correct collection in the vector store.
    *   It generates an embedding for the user's query and performs a similarity search within that collection to find the most relevant text chunks.
    *   These chunks are then compiled into a context, which is passed to the LLM along with the original query to generate a final, context-aware answer.

> **Endpoint Configuration: SaaS vs. Local**
> The embedding models used in step 3 are managed by the `LLMAgnosticClient`, which offers flexibility in endpoint configuration. This allows administrators to choose between data privacy and performance:
> *   **External SaaS Endpoints:** The framework can be configured to use commercial embedding services like **OpenAI** or **Azure OpenAI**. This approach provides access to powerful, state-of-the-art models but requires sending the text chunks over the internet to a third-party service.
> *   **Local Endpoints:** For environments where data cannot leave the local network, the framework can be pointed to a self-hosted model served via a local endpoint (e.g., an **Ollama** server). This ensures maximum data privacy at the cost of requiring local computational resources for the model.

### 1.2. General Utilities

These tools provide the agent with basic functionalities for calculations and session management.

**Tools:**
- `get_current_datetime`
- `perform_calculation`
- `store_note_in_session`
- `retrieve_session_notes`

#### How Does It Work?

**Example User Query:** "What is (100 / 5) + 3? Also, please remind me to follow up with Dr. Smith next week."

1.  **Tool Selection:** The `ScientificWorkflowAgent` parses the user's request and identifies the need for two separate tools: `perform_calculation` for the math expression and `store_note_in_session` for the reminder.

2.  **Calculation:**
    *   The agent calls `perform_calculation` with the expression `(100 / 5) + 3`.
    *   The tool evaluates the expression and returns the result.

3.  **Note-Taking:**
    *   The agent then calls `store_note_in_session` with the text "Follow up with Dr. Smith next week."
    *   The tool stores this note in the user's session data, ensuring it can be retrieved later.

### 1.3. Bioinformatics Tools

This suite of tools allows the agent to interact with major bioinformatics databases and services.

**Tools:**
- UniProt, BLAST, PubChem, AlphaFold

#### How Does It Work?

**Example User Query:** "Find the UniProt entry for 'APP_HUMAN', then run a BLAST search with its sequence."

1.  **UniProt Query:**
    *   The agent first calls the `query_uniprot_by_accession` tool with the ID `APP_HUMAN`.
    *   The tool sends a request to the UniProt API, retrieves the protein's data, and extracts key information, including its amino acid sequence.

2.  **BLAST Search:**
    *   The agent then takes the sequence obtained from UniProt and calls the `perform_blastp_search_biopython` tool.
    *   This tool uses Biopython's `NCBIWWW.qblast` function to perform a BLAST search against the NCBI database, returning a list of homologous sequences.

### 1.4. Web Search Tools

These tools enable the agent to search the web for information, using either a robust API or a scraping-based fallback.

**Tools:**
- `perform_web_search_api`
- `perform_web_search_scraping`

#### How Does It Work?

**Example User Query:** "What are the latest advancements in CRISPR technology?"

1.  **API-First Approach:** The agent prioritizes the `perform_web_search_api` tool. This tool connects to an **external SaaS endpoint** (the Brave Search API), which requires an active internet connection and a configured API key. It provides reliable, structured results and is the preferred method for web searches.

2.  **Scraping Fallback:** If the API is unavailable or fails, the agent can fall back to `perform_web_search_scraping`. This tool also connects to an external endpoint (a public search engine like DuckDuckGo) but does so by simulating a user in a web browser, which can be less reliable and slower than a dedicated API.

### 1.5. Multi-Agent Orchestration

These tools allow the main agent to act as a manager, creating and supervising a team of specialized AI agents to accomplish complex, multi-step tasks. This is ideal for workflows that require different areas of expertise or parallel lines of investigation.

#### How Does It Work?

**Example User Query:** "Draft a multi-phase plan for a competitive analysis of a new drug candidate targeting protein P01112. Create a team with a bioinformatician and a chemist to analyze its sequence, find existing patents, and identify similar compounds."

This workflow operates in the **router** execution mode, which involves a **Planner Agent** to create a strategy and a **Supervisor Agent** to execute it.

**Phase 1: Session and Plan Creation**

1.  **Team Assembly:** The main `ScientificWorkflowAgent` (acting as the user's primary interface) receives the query. It recognizes the need for a specialized team and calls the `CreateMultiAgentSession` tool with the roles `{'roles': ['bioinformatician', 'chemist']}`. This creates a new, isolated session for the team and returns a `multi_agent_session_id`.

2.  **Plan Generation:** The main agent then calls the `GeneratePlanForMultiAgentTask` tool, providing the `multi_agent_session_id` and the user's task. Inside the multi-agent framework, a dedicated **Planner Agent** creates a detailed, step-by-step plan. The plan is constrained to the tools available to the worker agents (e.g., `QueryUniProt`, `ExecuteCode`, etc.).

3.  **User Approval:** The generated plan is returned to the main agent, who **must present it to the user for approval**. The user can approve the plan as-is or suggest edits.

**Example Generated Plan:**

*   **Phase 1: Target Analysis (Bioinformatician)**
    1.  **Task:** Retrieve the amino acid sequence and key annotations for protein P01112.
        *   **Tool:** `QueryUniProt` with `accession_id='P01112'`.
    2.  **Task:** Perform a protein BLAST search to find homologous sequences.
        *   **Tool:** `PerformProteinBlastSearchBiopython` with the sequence from the previous step.
    3.  **Task:** Analyze the BLAST results to identify highly similar proteins. Encode the list of top 5 accessions as a Base64 string for the next phase.
        *   **Tool:** `ExecuteCode` with a Python script to parse the BLAST output.

*   **Phase 2: Chemical and Patent Analysis (Chemist)**
    1.  **Task:** Decode the Base64 string of homologous protein accessions from the previous phase. For each accession, search the web for associated patents.
        *   **Tool:** `ExecuteCode` (to decode) followed by multiple calls to `PerformWebSearchAPI`.
    2.  **Task:** Search PubChem for compounds known to interact with the primary target P01112.
        *   **Tool:** `SearchPubChemByName` with query `P01112 interaction`.
    3.  **Task:** Retrieve detailed properties for the top 3 compounds found.
        *   **Tool:** `GetPubChemCompoundProperties` for each CID.
    4.  **Task:** Consolidate all findings (patent links, compound details) into a final summary. Encode the summary as a Base64 string.
        *   **Tool:** `ExecuteCode`.

**Phase 2: Execution and State Management**

4.  **Plan Execution:** Once the user approves the plan, the main agent calls `ExecuteApprovedPlan`. The **Supervisor Agent** takes over and begins executing the plan step-by-step, delegating each task to the correct worker agent (`bioinformatician` or `chemist`).

5.  **State Passing with Base64:** To pass complex data between steps, the agents use Base64 encoding. For instance, at the end of Phase 1, the bioinformatician's Python script will not only analyze the BLAST results but also encode the list of protein accessions into a Base64 string and print it to `stdout`. The Supervisor captures this output.

6.  **Decoding and Continuing:** In Phase 2, the Supervisor passes this Base64 string as an argument to the chemist's first task. The chemist's Python script will first decode the string to retrieve the list of accessions before proceeding with the patent search.

**Example Python Script with Dependency Installation and Base64 Encoding:**

```python
# Step 1: Install dependencies in the sandbox
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    install('pandas')
except Exception as e:
    print(f"Error installing pandas: {e}", file=sys.stderr)
    sys.exit(1)

# Step 2: Import libraries and perform the task
import pandas as pd
import base64
import json

# Assume blast_results_json is a JSON string passed as an argument
blast_results_json = '''{"results": [{"accession": "P12345"}, {"accession": "P67890"}]}'''
data = json.loads(blast_results_json)

df = pd.DataFrame(data['results'])
top_accessions = df['accession'].head(5).tolist()

# Step 3: Encode the output as Base64 and print it
accessions_json = json.dumps(top_accessions)
encoded_accessions = base64.b64encode(accessions_json.encode('utf-8')).decode('utf-8')

print(f"Encoded Accessions: {encoded_accessions}")
```

**Phase 3: Cleanup**

7.  **Final Report:** The Supervisor synthesizes all results into a comprehensive report and returns it to the main agent.
8.  **Session Termination:** Finally, the main agent calls `TerminateMultiAgentSession` to clean up the resources used by the team.

## 2. HPC MCP Server (`hpc_mcp_server`)

This server provides tools for computationally intensive tasks that are offloaded to a high-performance computing (HPC) environment.

### 2.1. Nextflow BLAST Pipeline

**Tool:** `run_nextflow_blast_pipeline`

#### How Does It Work?

**Example User Query:** "Run a BLAST search for this large sequence file against a custom database."

1.  **Offloading to HPC:** The agent calls `run_nextflow_blast_pipeline`, which is handled by the HPC MCP server.
2.  **Pipeline Execution:** The tool executes a Nextflow pipeline (`blast_pipeline.nf`) that runs the BLAST search in a containerized environment, ensuring scalability and reproducibility.

### 2.2. Video Transcription Pipeline

**Tool:** `run_video_transcription_pipeline`

#### How Does It Work?

**Example User Query:** "Transcribe the audio from this YouTube video and give me a summary."

1.  **Video Processing:** The agent calls `run_video_transcription_pipeline` with the video URL.
2.  **Transcription and Summarization:** The tool uses a Nextflow pipeline to download the video, extract the audio, transcribe it using Whisper, and then generate a summary with an LLM. The full transcript is also indexed for future RAG queries.

### 2.3. GitHub Secret Scanning

**Tool:** `scan_github_repository_with_gitxray`

#### How Does It Work?

**Example User Query:** "Scan the repository at `https://github.com/example/repo` for any exposed secrets."

1.  **Security Scan:** The agent calls `scan_github_repository_with_gitxray`.
2.  **GitXRay Execution:** The tool runs a Nextflow pipeline that uses GitXRay to scan the entire history of the specified GitHub repository for secrets and other sensitive data, returning a detailed report.

> **Endpoint Configuration: SaaS vs. Local**
> While the `gitxray` scanning tool itself runs locally within the HPC server's containerized environment, it must connect to an **external SaaS endpoint** (`github.com`) to clone the target repository. This means the HPC server requires outbound internet access to perform the scan.

## 3. Sandbox MCP Server (`sandbox_mcp_server`)

This server provides a secure environment for executing code, preventing any potential harm to the host system.

### 3.1. Code Execution

**Tool:** `execute_code`

#### How Does It Work?

**Example User Query:** "Calculate the standard deviation for the following list of numbers: [1, 2, 3, 4, 5] and plot the distribution."

1.  **Code Generation:** The agent generates a Python script to perform the calculation and create a plot using a library like `matplotlib` or `seaborn`.

2.  **Sandboxed Execution:** It then calls the `execute_code` tool with the script and sets the `generate_plot` parameter to `True`. The Sandbox MCP Server runs the code in an isolated Docker container using the `llm-sandbox` library, which has networking disabled and a strict execution timeout.

3.  **Result Capturing:** The tool captures the `stdout`, `stderr`, and any generated plots from the sandbox. Plots are returned as Base64 encoded images, which the agent can then display to the user.
