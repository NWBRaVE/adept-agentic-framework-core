import os
import asyncio
import json
import operator
from typing import List, Dict, Any, Optional, TypedDict, Annotated, Sequence

# Langchain and LangGraph imports
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage, ToolMessage
from langchain_core.callbacks.base import BaseCallbackHandler, AsyncCallbackHandler
from langgraph.prebuilt import create_react_agent

# Local module imports for the new architecture
from .graph_builder import AgentGraphBuilder
from .mcp_langchain_tools import (
    get_mcp_query_csv_tool_langchain,
    get_mcp_perform_calculation_tool_langchain,
    get_mcp_store_note_tool_langchain,
    get_mcp_retrieve_notes_tool_langchain,
    get_mcp_query_uniprot_tool_langchain,
    get_mcp_web_search_api_tool_langchain, # Preferred web search tool using API service
    get_mcp_web_search_scraping_tool_langchain, # Backup web search tool using scraping
    get_mcp_blastp_biopython_tool_langchain, # Import Biopython BLASTP tool
    get_mcp_blastn_biopython_tool_langchain,  # Import Biopython BLASTN tool
    get_mcp_search_pubchem_by_query_tool_langchain, # Import PubChem search tool factory
    get_mcp_search_pubchem_by_name_tool_langchain, # Import PubChem search tool factory
    get_mcp_get_pubchem_compound_properties_tool_langchain, # Import PubChem properties tool factory
    get_mcp_list_uploaded_files_tool_langchain, # Import new tool for listing files
    get_mcp_alphafold_prediction_tool_langchain, # Import AlphaFold prediction tool
    get_mcp_query_stored_alphafold_tool_langchain, # Import AlphaFold RAG query tool
    get_mcp_run_nextflow_blast_tool_langchain, # Import HPC Nextflow BLAST tool
    get_mcp_run_video_transcription_tool_langchain, # Import HPC Video Transcription tool
    get_mcp_execute_code_tool_langchain, # Import Sandbox Code Execution tool
    get_mcp_gitxray_scan_tool_langchain, # Import HPC GitXRay Scan tool
    get_mcp_create_multi_agent_session_tool_langchain, # Multi-agent tool
    get_mcp_generate_plan_for_multi_agent_task_tool_langchain, # Multi-agent tool
    get_mcp_generate_and_review_plan_for_multi_agent_task_tool_langchain, # Multi-agent tool
    get_mcp_execute_approved_plan_tool_langchain, # Multi-agent tool
    get_mcp_update_pending_plan_tool_langchain, # Multi-agent tool
    get_mcp_terminate_multi_agent_session_tool_langchain, # Multi-agent tool
    get_mcp_list_active_multi_agent_sessions_tool_langchain, # Multi-agent tool
)
from .agent_state import AgentState
from ..core.llm_agnostic_layer import LLMAgnosticClient
from ..logger_config import get_logger

logger = get_logger(__name__)

# Simple way to disable redis-base checkpointing (or any custom checkpointer)
try:
    from .custom_checkpointer import get_redis_checkpointer
except Exception as custom_checkpointer_err:
    get_redis_checkpointer = None
    logger.warning(f"Custom checkpointer DISABLED: {custom_checkpointer_err}")
    

class ScientificWorkflowAgent:
    """The primary orchestrator for all scientific workflows.

    This class acts as the central brain, responsible for interpreting user requests,
    selecting and executing tools, and managing the overall workflow. It is designed
    to be highly configurable, allowing for dynamic selection of its core execution
    logic and persistence layer.

    Design Discussion: Custom StateGraph vs. create_react_agent
    -----------------------------------------------------------
    This agent can be configured to use one of two underlying graph-building strategies.
    The choice, controlled by the `USE_SPLIT_STREAM_GRAPH` environment variable, has
    significant implications for flexibility and control.

    **1. `create_react_agent` (Default Mode):**
    This is a high-level utility from LangGraph that quickly creates a standard ReAct
    (Reasoning + Acting) agent.
    -   **Pros:** High reliability for common workflows, simple to implement, and
        maintained by the LangChain team.
    -   **Cons:** It is a "black box." The internal logic is hidden, making it difficult
        to customize the data flow or debug non-standard interactions. It was not
        suitable for implementing the "split-stream" strategy for tool outputs.

    **2. Custom `StateGraph` (Split-Stream Mode):**
    This approach involves manually defining every node and edge in the agent's workflow
    graph using `langgraph.graph.StateGraph`.
    -   **Pros:**
        -   **Maximum Control:** This is its greatest strength. It allows for complete
          authority over the data flow, which was essential for creating a custom tool
          node to intercept tool outputs, sanitize them for the LLM, and store the full
          version in the agent's state (the "split-stream" strategy).
        -   **Transparency & Extensibility:** The logic is explicit, making it easier to
          debug and extend with new behaviors like human-in-the-loop steps or
          self-correction loops.
    -   **Cons:** It is more verbose and requires a deeper understanding of LangGraph's
        state management and maintenance overhead.

    **Conclusion:**
    While `create_react_agent` is robust for standard workflows, the requirement to
    manage the LLM's context window by sanitizing large tool outputs (like plots)
    necessitated the custom `StateGraph` approach. We traded simplicity for the power
    and control needed to implement the advanced "split-stream" data handling strategy.

    In addition to its scientific tools, this agent is also equipped with the
    MCP Tool Proxy tools. This special toolset grants the agent the dynamic,
    runtime ability to connect to and utilize tools from other MCP servers.
    This allows the agent's capabilities to be expanded on-the-fly without
    requiring a server restart, providing a powerful mechanism for extensibility.

    This agent also supports Redis-backed checkpointing for state persistence,
    enabling long-running conversations and recovery from interruptions.
    """
    def __init__(self, mcp_session_id: Optional[str] = None, additional_tools: Optional[List] = None):
        self.mcp_session_id = mcp_session_id
        self.llm_agnostic_client = LLMAgnosticClient()
        self.llm, self.default_model_name = self.llm_agnostic_client.get_langchain_chat_model(
            llm_purpose="agent_main",
            model_name=os.getenv("LANGCHAIN_LLM_MODEL"),
            return_name=True
        )
        self.tools = [
            get_mcp_query_csv_tool_langchain(mcp_session_id=mcp_session_id),
            get_mcp_perform_calculation_tool_langchain(mcp_session_id=mcp_session_id),
            get_mcp_store_note_tool_langchain(mcp_session_id=mcp_session_id),
            get_mcp_retrieve_notes_tool_langchain(mcp_session_id=mcp_session_id),
            get_mcp_query_uniprot_tool_langchain(mcp_session_id=mcp_session_id),
            get_mcp_web_search_api_tool_langchain(mcp_session_id=mcp_session_id),
            get_mcp_web_search_scraping_tool_langchain(mcp_session_id=mcp_session_id),
            get_mcp_blastp_biopython_tool_langchain(mcp_session_id=mcp_session_id),
            get_mcp_blastn_biopython_tool_langchain(mcp_session_id=mcp_session_id),
            get_mcp_search_pubchem_by_name_tool_langchain(mcp_session_id=mcp_session_id),
            get_mcp_get_pubchem_compound_properties_tool_langchain(mcp_session_id=mcp_session_id),
            get_mcp_list_uploaded_files_tool_langchain(mcp_session_id=mcp_session_id),
            get_mcp_alphafold_prediction_tool_langchain(mcp_session_id=mcp_session_id),
            get_mcp_query_stored_alphafold_tool_langchain(mcp_session_id=mcp_session_id),
            get_mcp_run_nextflow_blast_tool_langchain(mcp_session_id=mcp_session_id),
            get_mcp_run_video_transcription_tool_langchain(mcp_session_id=mcp_session_id),
            get_mcp_execute_code_tool_langchain(mcp_session_id=mcp_session_id),
            get_mcp_gitxray_scan_tool_langchain(mcp_session_id=mcp_session_id),
            get_mcp_create_multi_agent_session_tool_langchain(mcp_session_id=mcp_session_id),
            get_mcp_generate_plan_for_multi_agent_task_tool_langchain(mcp_session_id=mcp_session_id),
            get_mcp_generate_and_review_plan_for_multi_agent_task_tool_langchain(mcp_session_id=mcp_session_id), # New tool
            get_mcp_execute_approved_plan_tool_langchain(mcp_session_id=mcp_session_id),
            get_mcp_update_pending_plan_tool_langchain(mcp_session_id=mcp_session_id),
            get_mcp_terminate_multi_agent_session_tool_langchain(mcp_session_id=mcp_session_id),
            get_mcp_list_active_multi_agent_sessions_tool_langchain(mcp_session_id=mcp_session_id),
        ]
        if additional_tools:
            self.tools.extend(additional_tools)
        # Construct the example JSON string safely to avoid f-string syntax errors.
        example_json = {
            "note_text": "my important note",
            "mcp_session_id": self.mcp_session_id or "YOUR_CURRENT_SESSION_ID"
        }
        # Use json.dumps to create a valid JSON string representation.
        # Replace double quotes with single quotes to match the desired output style.
        example_json_str = json.dumps(example_json).replace('"', "'")

        self.formatted_system_prompt = (
            "You are a helpful scientific workflow assistant. "
            "You have access to several tools to help answer user queries and perform tasks. "
            "These tools allow you to perform calculations, manage notes, "
            "perform a web search using a robust API (use PerformWebSearchAPI for this) or by scraping a website (use PerformWebSearchScraping as a fallback), query UniProt (QueryUniProt), "
            "search PubChem for chemical compounds (SearchPubChemByName), retrieve detailed compound properties by CID (GetPubChemCompoundProperties), "
            "fetch protein structure predictions from AlphaFold EBI (GetAlphaFoldPrediction), "
            "search your local cache of previously fetched AlphaFold predictions (SearchStoredAlphaFoldPredictions), "
            "perform protein (PerformProteinBlastSearchBiopython) and nucleotide (PerformNucleotideBlastSearchBiopython) sequence searches using Biopython (NCBI web service); "
            "run a BLAST search pipeline on a dedicated HPC server (RunNextflowBlastPipelineHPC) for potentially larger or custom database searches; "
            "and transcribe and summarize videos (RunVideoTranscriptionPipelineHPC) using a video URL or an absolute server-accessible file path (e.g., /app/data/uploaded_files/my_video.mp4). This tool also indexes the transcript, making it queryable via its returned file_id using QueryProcessedDocumentData. "
            "retrieve information from uploaded documents (CSVs, PDFs, DOCX, images, TXT, TEX) once they are processed and have a file_id (QueryProcessedDocumentData), list previously uploaded and processed files (ListUploadedFiles), "
            "execute arbitrary code in a secure sandbox (ExecuteCode) for languages like python, javascript, or shell. For Python, you can generate plots by setting the 'generate_plot' parameter to True. This can create static images (e.g., with matplotlib) or interactive HTML plots (e.g., with plotly). For Python plotting, ensure you explicitly create a figure object (e.g., 'fig, ax = plt.subplots()') and avoid calling 'plt.show()'. The figure object should be the last expression in your code block for it to be captured. Streamlit renders these figures using 'st.pyplot(fig)'. You can also specify a custom Docker image with the 'sandbox_image' parameter; and scan public GitHub repositories for secrets (ScanGithubRepositoryForSecrets). "
            "To use the multi-agent team for complex tasks, you MUST follow this exact procedure:\n"
            "1. Create a team with `CreateMultiAgentSession`. You will get a `multi_agent_session_id`.\n"
            "2. Create a plan for the team to execute. For complex tasks, you should use `GenerateAndReviewPlanForMultiAgentTask`. For simpler tasks, `GeneratePlanForMultiAgentTask` is sufficient. This tool will return a plan for your review. You MUST present the multi-agent session `multi_agent_session_id` and append it to all of your responses as a reference.\n"
            "3. You MUST present the complete plan to the user. Ask for their approval. They may approve it as-is or suggest edits.\n"
            "4. If the user provides edits, you MUST use the `UpdatePendingPlan` tool to apply them. If the user approves the plan without changes (e.g., says 'yes' or 'proceed'), you MUST NOT generate the plan again.\n"
            "5. Once the plan is approved (either original or edited), you MUST call `ExecuteApprovedPlan` with the correct `multi_agent_session_id` to start the execution.\n"
            "6. After the task is complete, use `TerminateMultiAgentSession` to clean up.\n"
            "You can use `ListActiveMultiAgentSessions` at any time to see available teams.\n"
            "If there are processed files avaiable, you can use the QueryProcessedCSVData tool to query processed files. "
            "When you use a tool that requires a session context (like storing notes or querying user-specific CSV data), "
            "you MUST include the 'mcp_session_id' in the tool's input parameters. "
            f"The current MCP session ID for your operations is: {self.mcp_session_id or 'N/A'}. "
            f"For example, if you use the 'StoreNoteInSession' tool, your input should be a JSON like: {example_json_str}. "
            "Always provide the `mcp_session_id` when the tool's description or input schema indicates it is needed for session context."
        )

        # --- Dynamic Graph and Checkpointer Setup ---
        self.use_split_stream = os.getenv("USE_SPLIT_STREAM_GRAPH", "false").lower() == "true"
        self.use_checkpointing = os.getenv("USE_CHECKPOINTING", "false").lower() == "true"

        # Setup persistence
        # Conditionally enable the checkpointer for the custom graph
        memory = None
        if self.use_checkpointing and get_redis_checkpointer is not None:
            logger.info("Persistence: Enabled (USE_CHECKPOINTING=true). Using Redis checkpointer.")
            memory = get_redis_checkpointer()
        else:
            logger.info("Persistence: Disabled (USE_CHECKPOINTING=false).")
        
        if self.use_split_stream:
            # --- Build the advanced, custom graph with the split-stream feature ---
            logger.info("Agent mode: Custom Graph (USE_SPLIT_STREAM_GRAPH=true)")
            graph_builder = AgentGraphBuilder(llm=self.llm, tools=self.tools)
            workflow = graph_builder.build()
                    
            self.runnable = workflow.compile(checkpointer=memory)
            log_msg = f"Custom LangGraph agent compiled {'with' if memory else 'without'} persistence."
            logger.info(log_msg)
        else:
            # --- Fallback to the simple, default create_react_agent utility ---
            logger.info("Agent mode: Default (USE_SPLIT_STREAM_GRAPH=false). Using create_react_agent.")
            
            self.runnable = create_react_agent(
                model=self.llm,
                tools=self.tools,
                prompt=SystemMessage(content=self.formatted_system_prompt),
                debug=True,
                checkpointer=memory
            )
            logger.info("Default LangGraph agent compiled successfully.")

    async def arun(self, user_input: str, chat_history: Optional[List] = None, callbacks: Optional[List] = None, recursion_limit: int = 25) -> Dict[str, Any]:
        current_chat_history: List[BaseMessage] = []
        if chat_history:
            for msg in chat_history:
                if msg.get("role") == "user":
                    current_chat_history.append(HumanMessage(content=msg.get("content","")))
                elif msg.get("role") == "assistant":
                    current_chat_history.append(AIMessage(content=msg.get("content","")))

        initial_messages = [SystemMessage(content=self.formatted_system_prompt)] + current_chat_history + [HumanMessage(content=user_input)]
        
        try:
            if self.use_split_stream:
                # --- Prepare inputs for the custom graph ---
                initial_messages = [SystemMessage(content=self.formatted_system_prompt)] + current_chat_history + [HumanMessage(content=user_input)]
                graph_input = {
                    "messages": initial_messages,
                    "chat_history": chat_history or [],
                    "full_tool_outputs": []
                }
                config = {
                    "recursion_limit": recursion_limit,
                    "callbacks": callbacks,
                    "configurable": {"thread_id": self.mcp_session_id}
                }
                logger.info(f"Running custom compiled LangGraph for session {self.mcp_session_id}...")
                final_state_result = await self.runnable.ainvoke(graph_input, config=config)
            else:
                # --- Prepare inputs for the default create_react_agent graph ---
                initial_messages_for_graph = current_chat_history + [HumanMessage(content=user_input)]
                graph_input = {"messages": initial_messages_for_graph}
                config = {"recursion_limit": recursion_limit, "callbacks": callbacks}
                logger.info(f"Running create_react_agent graph for session {self.mcp_session_id}...")
                final_state_result = await self.runnable.ainvoke(graph_input, config=config)
            
            if final_state_result and final_state_result.get('messages'):
                last_message_in_graph = final_state_result['messages'][-1]
                output_content = last_message_in_graph.content

                # Check the last tool message for plot URLs, regardless of which tool was called
                last_tool_message = next((msg for msg in reversed(final_state_result.get("messages", [])) if isinstance(msg, ToolMessage)), None)
                if last_tool_message:
                    try:
                        tool_output = json.loads(last_tool_message.content)
                        if isinstance(tool_output, dict) and tool_output.get("plots"):
                            plot_urls = [p.get('plot_url') for p in tool_output.get("plots", []) if p.get('plot_url')]
                            if plot_urls:
                                markdown_images = "\n".join([f"![Generated Plot]({url})" for url in plot_urls])
                                output_content = f"{markdown_images}\n\n{output_content}"
                                logger.info(f"Agent formatted plot URLs from tool '{last_tool_message.name}' into Markdown.")
                    except (json.JSONDecodeError, TypeError):
                        logger.warning(f"Could not parse tool output from '{last_tool_message.name}' as JSON: {last_tool_message.content}")

                if isinstance(last_message_in_graph, AIMessage) and last_message_in_graph.tool_calls:
                    logger.warning(f"LangGraph ended with AIMessage with tool_calls: {last_message_in_graph.tool_calls}")
                    output_content = f"Agent decided to call tools but ended: {last_message_in_graph.content} (Tools: {last_message_in_graph.tool_calls})"
                return {"output": output_content, "full_graph_state": final_state_result}
            else:
                logger.error(f"LangGraph execution for session {self.mcp_session_id} resulted in empty or malformed state.")
                return {"output": "LangGraph execution finished with no clear output.", "full_graph_state": final_state_result}
        except Exception as e:
            logger.error(f"Error running LangGraph for session {self.mcp_session_id}: {e}", exc_info=True)
            if callbacks:
                for cb_item in callbacks:
                    if cb_item and hasattr(cb_item, 'on_chain_error'):
                        try:
                            await cb_item.on_chain_error(e) if asyncio.iscoroutinefunction(cb_item.on_chain_error) else cb_item.on_chain_error(e)
                        except Exception as cb_err:
                            logger.error(f"Error in callback on_chain_error: {cb_err}")
            return {"output": f"An error occurred during LangGraph execution: {e}"}
