from typing import List, Dict, Any, Annotated, Sequence, TypedDict
import operator
from langchain_core.messages import BaseMessage

# This file defines the shared state object for the LangGraph agent.
# It is kept in a separate file to avoid circular imports between the
# agent and the graph builder.

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    chat_history: List[Dict[str, str]]
    full_tool_outputs: Annotated[List[Dict[str, Any]], operator.add]
