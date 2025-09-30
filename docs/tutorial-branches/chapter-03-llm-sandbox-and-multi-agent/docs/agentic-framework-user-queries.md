# Agentic Framework - User Test Queries

This document provides a comprehensive set of sample queries to test the functionality of the various tools integrated into the agentic framework. These queries are designed to be used in the Streamlit UI to validate the agent's reasoning, tool selection, and response generation capabilities.

---

## 1. Main MCP Server Tools

These tools are hosted on the primary MCP server and handle general-purpose tasks like data retrieval, web search, and document analysis.

### 1.1. Document RAG (Retrieval-Augmented Generation)

*   **Functionality**: Allows the agent to ingest various document types (PDF, DOCX, CSV, XLSX, images), index their content, and answer questions based on the information within them. This is a two-step process: first ingest, then query.
*   **Codebase References**:
    *   Tool Logic: `src/agentic_framework_pkg/mcp_server/tools/csv_rag_tool.py`
    *   Langchain Wrapper: `src/agentic_framework_pkg/scientific_workflow/mcp_langchain_tools.py`
*   **Sample Queries**:
    *   **Prerequisite**: User must first upload a document (e.g., `my_research_paper.pdf`) through the Streamlit UI's "Document RAG Tool Test" section and get a `file_id`.
    *   **Query 1 (Information Extraction)**: "Using the document with file_id '...', what is the main conclusion of the study?"
    *   **Query 2 (Specific Data Point)**: "From the document with file_id '...', what was the reported p-value for the primary endpoint?"
    *   **Query 3 (List Documents)**: "List all the documents that have been processed."
    *   **Query 4 (Delete Document)**: "Delete the document with file_id '...'."
*   **Expected Agent Behavior / Answer**:
    *   For queries 1 & 2, the agent should call the `QueryProcessedDocumentData` tool with the provided `file_id` and the user's question. It should then synthesize the retrieved text into a natural language answer.
    *   For query 3, the agent should call `ListProcessedDocuments` and return a formatted list of file IDs and their original names.
    *   For query 4, the agent should call `DeleteProcessedDocument` and confirm that the document has been removed.

### 1.2. Web Search

*   **Functionality**: Enables the agent to perform web searches to find up-to-date information, answer general knowledge questions, or find links to resources.
*   **Codebase References**:
    *   Tool Logic: `src/agentic_framework_pkg/mcp_server/tools/websearch_tool.py`
    *   Langchain Wrapper: `src/agentic_framework_pkg/scientific_workflow/mcp_langchain_tools.py`
*   **Sample Queries**:
    *   **Query 1 (Fact-finding)**: "Who is the current director of the NIH?"
    *   **Query 2 (Finding a resource)**: "What is the official website for the Python library 'pandas'?"
    *   **Query 3 (Recent information)**: "Find recent news articles about the use of AI in drug discovery."
*   **Expected Agent Behavior / Answer**:
    *   The agent will call the `PerformWebSearch` tool with a reformulated query (e.g., "current NIH director"). It will then parse the search results (titles, links, snippets) and provide a concise answer to the user.

### 1.3. UniProt Database Query

*   **Functionality**: Allows the agent to query the UniProt database to retrieve information about proteins, such as sequences, functions, and metadata, using an accession number, gene name, or keyword.
*   **Codebase References**:
    *   Tool Logic: `src/agentic_framework_pkg/mcp_server/tools/uniprot_tool.py`
    *   Langchain Wrapper: `src/agentic_framework_pkg/scientific_workflow/mcp_langchain_tools.py`
*   **Sample Queries**:
    *   **Query 1 (By Accession)**: "Get the FASTA sequence for the UniProt accession P01112."
    *   **Query 2 (By Gene Name)**: "Find information about the human KRAS protein in UniProt."
    *   **Query 3 (By Keyword)**: "Search UniProt for proteins related to 'photosynthesis' in spinach."
*   **Expected Agent Behavior / Answer**:
    *   The agent will call the `QueryUniProt` tool with the appropriate `query` and `query_type`. It will then format the returned data (e.g., sequence, function, organism) into a readable response.

### 1.4. PubChem Database Query

*   **Functionality**: Enables the agent to search the PubChem database for chemical compounds by name and retrieve their properties using a Compound ID (CID).
*   **Codebase References**:
    *   Tool Logic: `src/agentic_framework_pkg/mcp_server/tools/pubchem_tool.py`
    *   Langchain Wrapper: `src/agentic_framework_pkg/scientific_workflow/mcp_langchain_tools.py`
*   **Sample Queries**:
    *   **Query 1 (Search by Name)**: "Search PubChem for the chemical 'caffeine'."
    *   **Query 2 (Get Properties)**: "After finding the CID for caffeine, get its Molecular Weight and InChIKey." (This can be a follow-up question).
    *   **Query 3 (Combined)**: "Find the molecular formula for aspirin using PubChem."
*   **Expected Agent Behavior / Answer**:
    *   For query 1, the agent calls `SearchPubChemByName`. It should return a list of matching compounds and their CIDs.
    *   For query 2, the agent calls `GetPubChemCompoundProperties` with the CID from the previous step.
    *   For query 3, the agent will likely chain the two calls: first `SearchPubChemByName` to get the CID for "aspirin", then `GetPubChemCompoundProperties` to retrieve the `MolecularFormula`.

### 1.5. Local BLAST Search

*   **Functionality**: Performs local protein (BLASTP) or nucleotide (BLASTN) sequence similarity searches using Biopython. This is suitable for quick, on-demand searches against a small, pre-configured database.
*   **Codebase References**:
    *   Tool Logic: `src/agentic_framework_pkg/mcp_server/tools/blast_tool.py`
    *   Langchain Wrapper: `src/agentic_framework_pkg/scientific_workflow/mcp_langchain_tools.py`
*   **Sample Queries**:
    *   **Query 1 (Protein)**: "Perform a BLASTP search for the protein sequence 'MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHQYREQIKRVKDSDDVPMVLVGNKCDLAARTVESRQAQDLARSYGIPYIETSAKTRQGVEDAFYTLVREIRQH'."
    *   **Query 2 (Nucleotide)**: "Run a BLASTN search for the nucleotide sequence 'AGCTTTTCATTCTGACTGCAACGGGCAATATGTCTCTGTGTGGATTAAAAAAAGAGTGTCTGATAGCAGC'."
*   **Expected Agent Behavior / Answer**:
    *   The agent will call `PerformProteinBlastSearch` or `PerformNucleotideBlastSearch` with the provided sequence. It will return a summary of the top hits, including descriptions, scores, and e-values.

---

## 2. HPC MCP Server Tools

These tools are hosted on a separate server designed for resource-intensive computations.

### 2.1. Nextflow BLAST Search

*   **Functionality**: Offloads a large-scale BLAST search to a more powerful computational environment using a Nextflow pipeline. This is designed for situations where the local BLAST tool is insufficient.
*   **Codebase References**:
    *   Tool Logic: `src/agentic_framework_pkg/hpc_mcp_server/tools/nextflow_blast_tool.py`
*   **Sample Queries**:
    *   **Query 1**: "Run a large-scale Nextflow BLAST search for the protein sequence 'MTEYKLVVVG...'. Use the 'swissprot' database."
*   **Expected Agent Behavior / Answer**:
    *   The agent should recognize the request for a "large-scale" or "Nextflow" search and invoke the `RunNextflowBlast` tool on the HPC server. The tool will return a job ID or a message indicating the pipeline has started. The agent should inform the user that the job is running.

### 2.2. Video/Audio Processing for RAG

*   **Functionality**: Downloads a video or audio file from a URL, transcribes the content using Whisper, generates a summary, and indexes the text for RAG. This makes multimedia content searchable.
*   **Codebase References**:
    *   Tool Logic: `src/agentic_framework_pkg/hpc_mcp_server/tools/video_processing_tool.py`
*   **Sample Queries**:
    *   **Query 1**: "Process the video from this URL for RAG: [YouTube URL]. After it's done, tell me what the main topics discussed are."
*   **Expected Agent Behavior / Answer**:
    *   The agent will call the `ProcessVideoForRag` tool on the HPC server. This is a long-running task. The agent should inform the user that the process has started. Once complete, the agent will receive a `file_id` for the transcribed text. In the second part of the query, it will use this `file_id` with the `QueryProcessedDocumentData` tool to answer the user's question about the video's content.

---

## 3. Sandbox MCP Server Tools

This server provides a secure environment for executing arbitrary code.

### 3.1. Code Execution

*   **Functionality**: Executes arbitrary Python, JavaScript, or shell code in a secure, isolated sandbox environment. This is extremely powerful for dynamic calculations, data manipulation, or running small scripts.
*   **Codebase References**:
    *   Tool Logic: `src/agentic_framework_pkg/sandbox_mcp_server/tools/code_execution_tool.py`
*   **Sample Queries**:
    *   **Query 1 (Python Calculation)**: "Calculate the 50th Fibonacci number using Python."
    *   **Query 2 (Data Manipulation)**: "I have a list of numbers: [1, 5, 2, 8, 3]. Write Python code to sort this list and find the median."
    *   **Query 3 (Shell Command)**: "Use a shell command to count the number of words in the string 'hello world this is a test'."
*   **Expected Agent Behavior / Answer**:
    *   The agent will identify the need to run code, formulate the code snippet, and call the `ExecuteCode` tool with the code and the specified language (`python`, `javascript`, or `shell`). It will then return the `stdout` or `stderr` from the execution as the answer.

---

## 4. Multi-Agent Orchestration Tools

*   **Functionality**: A sophisticated suite of tools for managing "teams" of AI agents to solve complex, multi-step problems that require planning and supervised execution. This allows for a human-in-the-loop to review and approve a plan before it's carried out.
*   **Codebase References**:
    *   Tool Logic: `src/agentic_framework_pkg/mcp_server/tools/multi_agent_tool.py`
*   **Sample Queries (as a sequential workflow)**:
    *   **1. Create Session**: "Create a new multi-agent session for the goal: 'Analyze the KRAS protein by finding its sequence, running a BLAST search, and identifying known inhibitors'."
    *   **2. Generate Plan**: "Now, generate a plan for the session I just created."
    *   **3. Review Plan**: "Show me the current plan."
    *   **4. Approve and Execute Plan**: "The plan looks good. Approve and execute the approved plan."
    *   **5. List Sessions**: "List all active multi-agent sessions."
    *   **6. Terminate Session**: "Terminate the session for the KRAS protein analysis."
*   **Expected Agent Behavior / Answer**:
    *   The agent will call the corresponding tool for each step (`CreateMultiAgentSession`, `GeneratePlan`, etc.).
    *   It will provide the user with session IDs, the generated plan for review, status updates during execution, and confirmation of termination. This workflow is highly interactive and relies on the user guiding the agent through the process.
    *   Note: The implementation of the multi-agent feature is purely illustrative. There are no guarantees that the plan that the multi-agent group generate will execute to completion. Certain tools and/or webservices may throw errors depending on the availability of external services, resources of the development environment, etc. Moreover, AI-service providers may enforce stricter AI Safety Guardrails especially with sensitive topics such as elicit drugs, weapons, self-harm, etc. 

    For example, you may encounter this error from Microsoft's Azure OpenAI API guardrails:
    ```bash
raise self._make_status_error_from_response(err.response) from None
agentic_streamlit_app   | openai.BadRequestError: Error code: 400 - {'error': {'message': "The response was filtered due to the prompt triggering Azure OpenAI's content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766", 'type': None, 'param': 'prompt', 'code': 'content_filter', 'status': 400, 'innererror': {'code': 'ResponsibleAIPolicyViolation', 'content_filter_result': {'hate': {'filtered': False, 'severity': 'safe'}, 'jailbreak': {'filtered': False, 'detected': False}, 'self_harm': {'filtered': True, 'severity': 'high'}, 'sexual': {'filtered': False, 'severity': 'safe'}, 'violence': {'filtered': False, 'severity': 'safe'}}}}}
    ```

--- 

