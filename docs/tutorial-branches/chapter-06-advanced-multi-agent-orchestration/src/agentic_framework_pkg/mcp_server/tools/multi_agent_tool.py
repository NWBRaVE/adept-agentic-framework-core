"""
This module provides the MCP tools for creating and managing multi-agent teams.

Architectural Design & Key Concepts
====================================

1.  **Agent-as-a-Tool and Dynamic Tool Provisioning:**
    This pattern ensures that specialized worker agents are fully capable and compatible
    with the entire suite of the framework's tools. An agent's specialization comes from its
    system prompt ("You are a world-class expert..."), not from a limited toolset. This
    instructs the agent to reason and act within its role, selecting the most appropriate
    tools from the full toolkit it possesses.

2.  **Dual-Mode Orchestration: Router vs. Graph:**
    The framework supports two distinct orchestration patterns, which are chosen explicitly
    when a session is created.

    **A. Router Mode (Default):**
       This is a deterministic, plan-based approach suitable for tasks with a clear sequence.
       The main `ScientificWorkflowAgent` is guided by its system prompt to follow a strict
       lifecycle:
       1.  **Create:** The agent calls `create_multi_agent_session` with `execution_mode='router'`.
       2.  **Plan:** The agent uses its reasoning to select a planning tool.
           -   **Simple Plan:** For straightforward tasks, it calls `generate_plan_for_multi_agent_task`.
           -   **Enhanced Plan:** For complex tasks, it calls `generate_and_review_plan_for_multi_agent_task`,
               which involves an internal review cycle by the agent team before the plan is finalized.
       3.  **Approve:** The agent presents the plan to the user for approval.
       4.  **Execute:** Upon approval, the agent calls `execute_approved_plan_in_session`. The
          Supervisor agent then delegates each step of the plan to the appropriate worker.

    **B. Graph Mode (Advanced):**
       This is a more flexible, dynamic approach for non-linear problems where the path to a
       solution is not known in advance.
       1.  **Create:** The agent must be explicitly instructed to call `create_multi_agent_session`
          with the parameter `execution_mode='graph'`.
       2.  **Execute:** The agent calls `execute_task_in_graph_session`. The `supervisor_graph`
          then dynamically routes tasks between worker agents based on the evolving state of
          the problem, continuing until a solution is reached.

       **Developer-Facing Flow (`create_supervisor_graph`):**
       1.  **State Definition:** The graph's state is defined by the `AgentState` TypedDict,
           which tracks the list of messages, the next agent to act, the original task,
           and a count of turns for each agent to prevent loops.
       2.  **Worker Nodes:** For each worker agent, a dedicated `agent_node` is added to the
           graph. This node invokes the agent and updates the state with its response.
       3.  **Supervisor Node:** The central "supervisor" node is an LCEL chain that prompts
           the LLM with the current conversation history and a function definition (`route`).
           The LLM is forced to output a JSON object specifying the `next` worker to act
           (or "FINISH").
       4.  **Graph Construction:** A `StateGraph` is instantiated. Worker nodes and the
           supervisor node are added.
       5.  **Edge Definition:** Edges are defined to route the state from every worker node
           *back* to the supervisor node.
       6.  **Conditional Routing:** A conditional edge is added from the supervisor node. It
           inspects the `next` field in the state and routes the workflow to the
           corresponding worker's node or to the special `END` node.
       7.  **Compilation:** The final graph is compiled into a runnable that can be invoked
           with the initial task.

    The main agent's system prompt is intentionally focused on the `router` mode to ensure
    robust and predictable behavior for the most common workflows.
"""
import asyncio
import uuid
from typing import Dict, Any, Optional, List, Tuple, TypedDict

from fastmcp import FastMCP, Context
from ...logger_config import get_logger
from ...core.llm_agnostic_layer import LLMAgnosticClient

# LangGraph and LangChain imports
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent

# Import all the tool wrappers to give to the dynamic agents
from ...scientific_workflow.mcp_langchain_tools import (
    get_mcp_query_csv_tool_langchain,
    get_mcp_perform_calculation_tool_langchain,
    get_mcp_store_note_tool_langchain,
    get_mcp_retrieve_notes_tool_langchain,
    get_mcp_query_uniprot_tool_langchain,
    get_mcp_web_search_api_tool_langchain,
    get_mcp_web_search_scraping_tool_langchain,
    get_mcp_blastp_biopython_tool_langchain,
    get_mcp_blastn_biopython_tool_langchain,
    get_mcp_search_pubchem_by_name_tool_langchain,
    get_mcp_get_pubchem_compound_properties_tool_langchain,
    get_mcp_list_uploaded_files_tool_langchain,
    get_mcp_alphafold_prediction_tool_langchain,
    get_mcp_query_stored_alphafold_tool_langchain,
    get_mcp_run_nextflow_blast_tool_langchain,
    get_mcp_run_video_transcription_tool_langchain,
    get_mcp_execute_code_tool_langchain,
    get_mcp_gitxray_scan_tool_langchain
)

logger = get_logger(__name__)

# In-memory store for multi-agent sessions.
MULTI_AGENT_SESSIONS: Dict[str, Any] = {}

# Global LLM client instance
_llm_agnostic_client_instance: Optional[LLMAgnosticClient] = None

# --- Agent State and Graph Definition for Graph-based Supervisor ---

class AgentState(TypedDict):
    messages: List[BaseMessage]
    next: str
    task: str
    agent_turn_count: Dict[str, int]

async def agent_node(state: AgentState, agent: Any, name: str) -> Dict[str, Any]:
    """Node that executes a worker agent."""
    state["agent_turn_count"][name] = state["agent_turn_count"].get(name, 0) + 1
    result = await agent.ainvoke(state)
    return {"messages": [AIMessage(content=result["messages"][-1].content, name=name)], "agent_turn_count": state["agent_turn_count"]}

def create_supervisor_graph(worker_agents: Dict[str, Any], supervisor_llm: Any) -> Any:
    """Creates a stateful graph for the supervisor."""
    members = list(worker_agents.keys())
    system_prompt = (
        "You are a supervisor tasked with managing a conversation between the following workers: {members}. "
        "Given the user's request, determine which worker should act next. Each worker will perform a task and respond with their results and status. "
        "To prevent infinite loops, you have access to `agent_turn_count`, which tracks how many times each agent has acted. If you see a high count for an agent without progress, consider routing to a different agent or finishing the task. "
        "When the user's request is fully addressed, respond with the word FINISH."
    ).format(members=", ".join(members))
    
    options = ["FINISH"] + members
    function_def = {
        "name": "route",
        "description": "Select the next worker to act.",
        "parameters": {
            "title": "routeSchema",
            "type": "object",
            "properties": {"next": {"title": "Next", "anyOf": [{"enum": options}]}},
            "required": ["next"],
        },
    }
    
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt)]
    )
    
    supervisor_chain = (
        prompt
        | supervisor_llm.with_structured_output(function_def, include_raw=False)
    )

    workflow = StateGraph(AgentState)
    for name, agent in worker_agents.items():
        workflow.add_node(name, lambda state, agent=agent, name=name: asyncio.run(agent_node(state, agent, name)))

    workflow.add_node("supervisor", supervisor_chain)

    for member in members:
        workflow.add_edge(member, "supervisor")

    conditional_map = {k: k for k in members}
    conditional_map["FINISH"] = END
    workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)
    workflow.set_entry_point("supervisor")

    return workflow.compile()

# --- Tool and Agent Creation ---

def get_all_mcp_tools(session_id: str) -> List[BaseTool]:
    """Factory function to get a list of all available MCP tools."""
    tool_factories = [
        get_mcp_query_csv_tool_langchain, get_mcp_perform_calculation_tool_langchain,
        get_mcp_store_note_tool_langchain, get_mcp_retrieve_notes_tool_langchain,
        get_mcp_query_uniprot_tool_langchain, get_mcp_web_search_api_tool_langchain,
        get_mcp_web_search_scraping_tool_langchain,
        get_mcp_blastp_biopython_tool_langchain, get_mcp_blastn_biopython_tool_langchain,
        get_mcp_search_pubchem_by_name_tool_langchain, get_mcp_get_pubchem_compound_properties_tool_langchain,
        get_mcp_list_uploaded_files_tool_langchain, get_mcp_alphafold_prediction_tool_langchain,
        get_mcp_query_stored_alphafold_tool_langchain, get_mcp_run_nextflow_blast_tool_langchain,
        get_mcp_run_video_transcription_tool_langchain, get_mcp_execute_code_tool_langchain,
        get_mcp_gitxray_scan_tool_langchain
    ]
    return [factory(mcp_session_id=session_id) for factory in tool_factories]

def create_worker_agent(role: str, session_id: str) -> Tuple[Any, str]:
    """Creates a worker agent with a specific role and access to all tools."""
    llm = _llm_agnostic_client_instance.get_langchain_chat_model(llm_purpose="agent_worker")
    tools = get_all_mcp_tools(session_id)
    prompt = SystemMessage(content=f"You are a world-class expert {role}. You must use the provided tools to complete the tasks assigned to you. Do not make up information. Fulfill the task to the best of your ability.")
    agent_graph = create_react_agent(model=llm, tools=tools, prompt=prompt)
    return agent_graph, role

class AgentRunnerTool(BaseTool):
    """A tool that allows the supervisor to run a worker agent."""
    agent_executor: Any
    name: str
    description: str

    def _run(self, task: str) -> str:
        """Runs the agent with the given task synchronously."""
        try:
            return asyncio.run(self._arun(task))
        except RuntimeError as e:
            if "cannot be called when another loop is running" in str(e):
                logger.warning("Asyncio loop issue in AgentRunnerTool._run, trying nest_asyncio.")
                import nest_asyncio
                nest_asyncio.apply()
                return asyncio.run(self._arun(task))
            raise e

    async def _arun(self, task: str) -> str:
        """
        Runs the agent with the given task, streaming its execution steps to build a
        detailed report for the supervisor.
        """
        logger.info(f"Supervisor delegating task to {self.name}: {task}")
        try:
            final_state = {}
            # Use astream to get the full history. We iterate through the stream and only
            # care about the final state, which contains the complete message history.
            async for state in self.agent_executor.astream({"messages": [HumanMessage(content=task)]}):
                final_state = state
            
            messages = final_state.get("messages", [])
            if not messages or len(messages) <= 1:
                return f"The {self.name} agent produced no meaningful output for the task."

            report_parts = [f"Execution trace for worker agent '{self.name}':"]
            last_tool_name = "unknown_tool"
            for msg in messages[1:]: # Skip initial HumanMessage
                if isinstance(msg, AIMessage):
                    if msg.content:
                        thought = msg.content.replace('\n', ' ').strip()
                        if thought:
                            report_parts.append(f"  - Thought: {thought}")
                    if msg.tool_calls:
                        for tc in msg.tool_calls:
                            last_tool_name = tc.get('name', 'unknown_tool')
                            report_parts.append(f"  - Action: Calling tool `{last_tool_name}` with arguments `{tc.get('args', {})}`.")
                elif isinstance(msg, ToolMessage):
                    observation = str(msg.content).replace('\n', ' ').strip()
                    if len(observation) > 300: # Truncate long observations for clarity
                        observation = observation[:300] + "..."
                    report_parts.append(f"  - Observation from `{last_tool_name}`: {observation}")

            final_answer_msg = messages[-1]
            if isinstance(final_answer_msg, AIMessage) and not final_answer_msg.tool_calls:
                 report_parts.append(f"\nFinal Answer from worker: {final_answer_msg.content}")
            
            return "\n".join(report_parts)
        except Exception as e:
            # If the worker agent fails, report the failure back to the supervisor.
            logger.error(f"Worker agent {self.name} failed to execute task '{task[:50]}...'. Error: {e}", exc_info=True)
            return f"The {self.name} agent FAILED to complete the task. Error: {e}"

def create_planner_agent(roles: List[str]) -> Any:
    """Creates the planner agent for the router-based approach."""
    llm = _llm_agnostic_client_instance.get_langchain_chat_model(llm_purpose="agent_planner")
    role_descriptions = ", ".join(roles)
    planner_prompt = (
        "You are an expert planner. Your job is to create a detailed, step-by-step plan to answer a user's request. "
        "You have a team of experts available with the following roles: {roles}. "
        "For each step in the plan, specify which expert role should perform the task. "
        "If you are given feedback on a plan, you must refine the plan to address the feedback. "
        "The plan will be executed by a supervisor agent. Do not try to execute the plan yourself or generate a final answer. "
        "Your only output should be the plan."
    ).format(roles=role_descriptions)
    return create_react_agent(model=llm, tools=[], prompt=SystemMessage(content=planner_prompt))

def create_router_supervisor_agent(worker_agents: List[Tuple[Any, str]], session_id: str) -> Any:
    """Creates the supervisor agent for the router-based approach."""
    llm = _llm_agnostic_client_instance.get_langchain_chat_model(llm_purpose="agent_supervisor")
    worker_tools = [
        AgentRunnerTool(
            name=f"run_{role.lower().replace(' ', '_')}_agent",
            description=f"Use this tool to delegate a specific task to the {role} expert. Provide a clear and complete description of the task for the expert.",
            agent_executor=executor
        ) for executor, role in worker_agents
    ]
    system_prompt = (
        "You are a supervisor agent. Your job is to orchestrate a team of expert agents to complete a user's request. "
        "You have two primary modes: plan execution and plan review. "
        "If you are given a plan to execute, you must delegate each step to the appropriate expert agent using your tools. Do not deviate from the plan. After executing all steps, synthesize the results into a single, comprehensive final report. "
        "If you are asked to review a plan, you must delegate the review task to each of the worker agents to get their feedback. Synthesize their feedback into a single, coherent review. "
        "If a worker agent returns a message indicating it FAILED, you must acknowledge this failure. "
        "Analyze the error message. If the error suggests missing information that the user could provide (e.g., an ambiguous term, a missing file ID), "
        "your final report should clearly state the failure and ask the user for the specific information needed to resolve the issue. "
        "Otherwise, simply report the failure as part of the final results. "
        "Do not use your own knowledge; rely solely on the outputs of the worker agents."
    )
    return create_react_agent(model=llm, tools=worker_tools, prompt=SystemMessage(content=system_prompt))

def register_tools(mcp: FastMCP, llm_client: LLMAgnosticClient):
    global _llm_agnostic_client_instance
    _llm_agnostic_client_instance = llm_client

    @mcp.tool()
    async def create_multi_agent_session(ctx: Context, roles: List[str], mcp_session_id: str, execution_mode: str = "router") -> Dict[str, Any]:
        """
        Creates a new multi-agent session.
        - mcp_session_id: The main session ID to which this multi-agent session belongs.
        - execution_mode: 'router' for a planner/supervisor model, 'graph' for a dynamic supervisor graph.
        """
        session_id = f"multi-agent-{uuid.uuid4()}"
        await ctx.info(f"Creating new {execution_mode} session {session_id} for roles: {roles}")

        try:
            worker_agents_list = [create_worker_agent(role, mcp_session_id) for role in roles]
            
            session_data = {
                "roles": roles,
                "execution_mode": execution_mode,
                "chat_history": []
            }

            if execution_mode == "router":
                session_data["supervisor"] = create_router_supervisor_agent(worker_agents_list, mcp_session_id)
                session_data["planner"] = create_planner_agent(roles)
            elif execution_mode == "graph":
                worker_agents_dict = {role: agent for agent, role in worker_agents_list}
                supervisor_llm = _llm_agnostic_client_instance.get_langchain_chat_model(llm_purpose="agent_supervisor")
                session_data["supervisor_graph"] = create_supervisor_graph(worker_agents_dict, supervisor_llm)
            else:
                raise ValueError("Invalid execution_mode. Must be 'router' or 'graph'.")

            MULTI_AGENT_SESSIONS[session_id] = session_data
            await ctx.info(f"Session {session_id} created successfully.")
            return {"status": "success", "multi_agent_session_id": session_id, "message": f"Successfully created {execution_mode} session."}
        except Exception as e:
            logger.error(f"Failed to create session: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    @mcp.tool()
    async def execute_task_in_graph_session(ctx: Context, multi_agent_session_id: str, task: str, mcp_session_id: Optional[str] = None) -> Dict[str, Any]:
        """Executes a task in a 'graph' mode multi-agent session."""
        session = MULTI_AGENT_SESSIONS.get(multi_agent_session_id)
        if not session or session.get("execution_mode") != "graph":
            return {"status": "error", "message": "Session not found or is not a graph-based session."}

        await ctx.info(f"Executing task in graph session {multi_agent_session_id}: {task}")
        graph = session["supervisor_graph"]
        initial_state = {"messages": [HumanMessage(content=task)], "task": task, "agent_turn_count": {}}
        
        try:
            final_state = await graph.ainvoke(initial_state, {"recursion_limit": 100})
            final_response = final_state["messages"][-1].content
            session["chat_history"].extend([HumanMessage(content=task), AIMessage(content=final_response)])
            return {"status": "success", "report": final_response}
        except Exception as e:
            logger.error(f"Error in graph execution for session {multi_agent_session_id}: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    @mcp.tool()
    async def generate_plan_for_multi_agent_task(ctx: Context, multi_agent_session_id: str, task: str, mcp_session_id: Optional[str] = None) -> Dict[str, Any]:
        """Generates a plan for a 'router' mode multi-agent session."""
        session = MULTI_AGENT_SESSIONS.get(multi_agent_session_id)
        if not session or session.get("execution_mode") != "router":
            return {"status": "error", "message": "Session not found or is not a router-based session."}
        
        planner = session["planner"]
        chat_history = session.get("chat_history", [])
        try:
            planner_messages = chat_history + [HumanMessage(content=task)]
            plan_result = await planner.ainvoke({"messages": planner_messages})
            generated_plan = plan_result.get("messages", [])[-1].content
            session["pending_plan"] = {"task": task, "plan": generated_plan}
            return {"status": "plan_generated_for_approval", "plan": generated_plan, "message": "A plan has been generated. You MUST now show the user the following plan for their review and approval. Upon user approval, you MUST call the 'ExecuteApprovedPlan' tool. Do not ask the user any other questions."}
        except Exception as e:
            logger.error(f"Error during plan generation: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}


    @mcp.tool()
    async def generate_and_review_plan_for_multi_agent_task(ctx: Context, multi_agent_session_id: str, task: str, mcp_session_id: Optional[str] = None) -> Dict[str, Any]:
        """Generates a plan, has it reviewed by the agent team, and returns the refined plan for user approval."""
        session = MULTI_AGENT_SESSIONS.get(multi_agent_session_id)
        if not session or session.get("execution_mode") != "router":
            return {"status": "error", "message": "Session not found or is not a router-based session."}

        planner = session["planner"]
        supervisor = session["supervisor"]
        chat_history = session.get("chat_history", [])

        try:
            # 1. Generate initial plan
            await ctx.info("Generating initial plan...")
            planner_messages = chat_history + [HumanMessage(content=task)]
            plan_result = await planner.ainvoke({"messages": planner_messages})
            initial_plan = plan_result.get("messages", [])[-1].content

            # 2. Review the plan
            await ctx.info("Reviewing the plan with the supervisor and team...")
            review_task = f"Please review the following plan and provide feedback for refinement:\n\n{initial_plan}"
            
            # The supervisor will use its worker agents to review the plan.
            review_result = await supervisor.ainvoke({"messages": chat_history + [HumanMessage(content=review_task)]})
            review_feedback = review_result.get("messages", [])[-1].content

            # 3. Refine the plan
            await ctx.info("Refining the plan based on feedback...")
            refinement_task = f"Please refine the following plan based on the provided feedback.\n\nOriginal Plan:\n{initial_plan}\n\nFeedback:\n{review_feedback}"
            refinement_result = await planner.ainvoke({"messages": chat_history + [HumanMessage(content=refinement_task)]})
            refined_plan = refinement_result.get("messages", [])[-1].content

            session["pending_plan"] = {"task": task, "plan": refined_plan}
            return {"status": "plan_generated_for_approval", "plan": refined_plan, "message": "A refined plan has been generated and reviewed by the team. Please show the user the following plan for their review and approval. Upon user approval, you MUST call the 'ExecuteApprovedPlan' tool. Do not ask the user any other questions."}
        except Exception as e:
            logger.error(f"Error during plan generation and review: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}


    @mcp.tool()
    async def execute_approved_plan_in_session(ctx: Context, multi_agent_session_id: str, mcp_session_id: Optional[str] = None) -> Dict[str, Any]:
        """Executes the approved plan in a 'router' mode multi-agent session."""
        session = MULTI_AGENT_SESSIONS.get(multi_agent_session_id)
        if not session or session.get("execution_mode") != "router":
            return {"status": "error", "message": "Session not found or is not a router-based session."}
        if "pending_plan" not in session:
            return {"status": "error", "message": "No pending plan to execute. Please generate a plan first."}
        
        pending_plan = session.pop("pending_plan")
        task = pending_plan["task"]
        generated_plan = pending_plan["plan"]
        supervisor = session["supervisor"]
        chat_history = session.get("chat_history", [])
        try:
            supervisor_task_with_plan = (
                f"Here is the user's overall task: '{task}'\n\n"
                f"Here is the plan you must follow to achieve it:\n{generated_plan}\n\n"
                "Please execute this plan and provide a final report."
            )
            result = await supervisor.ainvoke({"messages": chat_history + [HumanMessage(content=supervisor_task_with_plan)]})
            final_response = result.get("messages", [])[-1].content
            session["chat_history"].extend([HumanMessage(content=task), AIMessage(content=final_response)])
            return {"status": "success", "report": final_response}
        except Exception as e:
            logger.error(f"Error during plan execution: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    @mcp.tool()
    async def update_pending_plan_in_session(ctx: Context, multi_agent_session_id: str, edited_plan: str, mcp_session_id: Optional[str] = None) -> Dict[str, Any]:
        """Updates the pending plan in a 'router' mode session."""
        session = MULTI_AGENT_SESSIONS.get(multi_agent_session_id)
        if not session or session.get("execution_mode") != "router":
            return {"status": "error", "message": "Session not found or is not a router-based session."}
        if "pending_plan" not in session:
            return {"status": "error", "message": "No pending plan to update. Please generate a plan first."}
        
        session["pending_plan"]["plan"] = edited_plan
        return {"status": "success", "message": "The pending plan has been successfully updated. You can now execute it.", "updated_plan": edited_plan}

    @mcp.tool()
    async def list_active_multi_agent_sessions(ctx: Context, mcp_session_id: Optional[str] = None) -> Dict[str, Any]:
        """Lists all active multi-agent sessions."""
        if not MULTI_AGENT_SESSIONS:
            return {"active_sessions": []}
        
        active_sessions_list = [
            {
                "multi_agent_session_id": session_id,
                "roles": data.get("roles", []),
                "execution_mode": data.get("execution_mode", "unknown"),
                "history_length": len(data.get("chat_history", []))
            }
            for session_id, data in MULTI_AGENT_SESSIONS.items()
        ]
        return {"active_sessions": active_sessions_list}

    @mcp.tool()
    async def terminate_multi_agent_session(ctx: Context, multi_agent_session_id: str, mcp_session_id: Optional[str] = None) -> Dict[str, Any]:
        """Terminates a multi-agent session."""
        if multi_agent_session_id in MULTI_AGENT_SESSIONS:
            del MULTI_AGENT_SESSIONS[multi_agent_session_id]
            return {"status": "success", "message": f"Session {multi_agent_session_id} terminated."}
        else:
            return {"status": "error", "message": "Session not found."}

    logger.info("Multi-Agent MCP tools registered.")