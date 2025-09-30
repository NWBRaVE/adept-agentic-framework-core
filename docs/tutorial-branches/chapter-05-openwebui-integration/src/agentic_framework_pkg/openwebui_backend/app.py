from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
import os
import uuid
import json
import asyncio
from typing import List, Dict, Any
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv # Import load_dotenv

# Load environment variables from .env file at the very beginning
load_dotenv() 

from agentic_framework_pkg.scientific_workflow.langchain_agent import ScientificWorkflowAgent
from agentic_framework_pkg.logger_config import get_logger

logger = get_logger(__name__)

app = FastAPI()

# CORS Configuration
# This allows OpenWebUI to make requests to this FastAPI app from different origins.
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:8080",
    "http://localhost:8081",
    "http://localhost:8083",
    "http://localhost:8902",
    "http://127.0.0.1",
    "http://127.0.0.1:8902",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8080",
    "http://127.0.0.1:8081",
    "http://127.0.0.1:8082",
    "http://host.docker.internal", # Important for Docker communication
    "http://host.docker.internal:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions: Dict[str, Any] = {}

def format_sse_chunk(data: Dict[str, Any]) -> str:
    return f"data: {json.dumps(data)}\n\n"

@app.get("/v1/models")
async def list_models():
    logger.info("Received request on /v1/models")
    models_data = {
        "object": "list",
        "data": [
            {
                "id": "agentic-framework-cot", # This ID will appear in OpenWebUI's model selector
                "object": "model",
                "created": int(os.path.getctime(__file__)),
                "owned_by": "your-organization",
                "permission": [
                    {
                        "id": "model-perm-xyz",
                        "object": "model_permission",
                        "created": int(os.path.getctime(__file__)),
                        "allow_create_engine": True,
                        "allow_sampling": True,
                        "allow_logprobs": True,
                        "allow_search_indices": False,
                        "allow_view": True,
                        "allow_fine_tuning": False,
                        "organization": "*",
                        "group": None,
                        "is_blocking": False,
                    }
                ],
                "root": "agentic-framework-cot",
                "parent": None,
            }
        ]
    }
    logger.info(f"Sending models response: {models_data}")
    return JSONResponse(content=models_data)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        body = await request.json()
        logger.info(f"Received request on /v1/chat/completions: {body}")

        messages = body.get("messages", [])
        if not messages:
            raise HTTPException(status_code=400, detail="No messages provided")

        user_message = messages[-1]["content"]
        chat_history_from_request = messages[:-1]

        conversation_id = body.get("conversation_id", str(uuid.uuid4()))
        stream_requested = body.get("stream", False)
        
        # --- Extract model name from the request body ---
        # The 'model' field is sent by Open WebUI and should match the 'id'
        # returned by your /v1/models endpoint (e.g., "agentic-framework-cot").
        model_name = body.get("model", "default-llm-model") # Provide a fallback if needed
        logger.info(f"Model requested: {model_name}")

        if conversation_id not in sessions:
            logger.info(f"New conversation started. Conversation ID: {conversation_id}")
            agent = ScientificWorkflowAgent(mcp_session_id=conversation_id)
            sessions[conversation_id] = {
                "agent": agent,
                "chat_history": []
            }
            # Override the model_name for this session
            model_name = agent.default_model_name if agent.default_model_name else model_name
            logger.info(f"Initialized agent with model (patched): {model_name}")
        else:
            session = sessions[conversation_id]
            agent = session["agent"]
            # If the model name *could* change mid-conversation, you might need to re-initialize
            # the agent or update its LLM here. For simplicity, we assume it's set once per session.
            # If the agent is stateful and its LLM can be updated, you'd do:
            # agent.update_llm_model(model_name) # Requires a method in ScientificWorkflowAgent

        agent_chat_history_for_run = chat_history_from_request

        logger.info(f"Running agent for conversation {conversation_id} with input: {user_message}")

        async def generate_response_chunks():
            yield format_sse_chunk({
                "id": f"chatcmpl-{uuid.uuid4()}",
                "object": "chat.completion.chunk",
                "created": int(os.path.getctime(__file__)),
                "model": model_name, # Use the model_name from the request
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant"},
                        "finish_reason": None,
                    }
                ]
            })

            full_agent_response = ""
            response_dict = await agent.arun(
                user_input=user_message,
                chat_history=agent_chat_history_for_run,
                callbacks=[],
            )
            agent_response = response_dict.get("output", "Agent did not provide a standard output.")
            
            for char in agent_response:
                yield format_sse_chunk({
                    "id": f"chatcmpl-{uuid.uuid4()}",
                    "object": "chat.completion.chunk",
                    "created": int(os.path.getctime(__file__)),
                    "model": model_name, # Use the model_name from the request
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": char},
                            "finish_reason": None,
                        }
                    ],
                })
                full_agent_response += char
                await asyncio.sleep(0.01)

            yield format_sse_chunk({
                "id": f"chatcmpl-{uuid.uuid4()}",
                "object": "chat.completion.chunk",
                "created": int(os.path.getctime(__file__)),
                "model": model_name, # Use the model_name from the request
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": len(full_agent_response.split()),
                    "total_tokens": len(user_message.split()) + len(full_agent_response.split()),
                },
            })
            yield "data: [DONE]\n\n"

        if stream_requested:
            return StreamingResponse(generate_response_chunks(), media_type="text/event-stream")
        else:
            response_dict = await agent.arun(
                user_input=user_message,
                chat_history=agent_chat_history_for_run,
                callbacks=[],
            )
            agent_response = response_dict.get("output", "Agent did not provide a standard output.")
            
            response_id = f"chatcmpl-{uuid.uuid4()}"
            response_data = {
                "id": response_id,
                "object": "chat.completion",
                "created": int(os.path.getctime(__file__)),
                "model": model_name, # Use the model_name from the request
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": agent_response,
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            }
            logger.info(f"Sending response: {response_data}")
            return JSONResponse(content=response_data)

    except Exception as e:
        logger.error(f"Error in /v1/chat/completions: {e}", exc_info=True)
        return JSONResponse(content={"detail": str(e)}, status_code=500)

if __name__ == "__main__":
    port = int(os.getenv("OPENWEBUI_PORT", "8081"))
    uvicorn.run(app, host="0.0.0.0", port=port)
