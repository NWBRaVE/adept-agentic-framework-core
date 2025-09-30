"""
This module provides a Redis-backed checkpointer for LangGraph agents using `langgraph-checkpoint-redis`.

Design Discussion: Persistence and Concurrency

The `langgraph-checkpoint-redis` library leverages `aioredis` to provide a robust, asynchronous,
and production-ready solution for persisting LangGraph agent states. This is crucial for
building scalable, multi-user agentic systems as it avoids the concurrency limitations
of file-based databases like SQLite.

Key Architectural Decisions:

1.  **Asynchronous Redis Client (`aioredis`):**
    -   The underlying `aioredis` client ensures that all Redis operations are non-blocking,
        preventing the main application event loop from being stalled. This is essential
        for high-performance asynchronous applications.

2.  **`RedisSaver` from `langgraph-checkpoint-redis`:**
    -   This library provides a direct and optimized integration with LangGraph's checkpointing
        mechanism, handling the complexities of state serialization and deserialization
        to and from Redis.

3.  **Data Sanitization (Custom Logic if needed):**
    -   While `langgraph-checkpoint-redis` handles core persistence, custom sanitization
        logic (e.g., stripping large `full_tool_outputs` fields) can still be implemented
        if necessary by extending `RedisSaver` or by processing the state before it's passed
        to the checkpointer. (Note: The previous `CustomRedisCheckpointer` was removed as
        `RedisSaver` is now used directly, but the concept of sanitization remains relevant
        for managing context window size).

This approach provides a scalable and maintainable solution for agent persistence,
fully leveraging asynchronous capabilities.
"""

import os
from typing import Dict, Any, Optional, Tuple
from langgraph_checkpoint_redis import RedisSaver

class CustomRedisCheckpointer(RedisSaver):
    """
    A custom Redis checkpointer that sanitizes the checkpoint before saving.
    It removes the 'full_tool_outputs' field to avoid persisting large,
    transient data like base64 plot strings to the database.
    """
    def put(self, config: dict, checkpoint: dict, metadata: dict) -> None:
        # Before writing to Redis, remove the large, transient data.
        checkpoint_copy = checkpoint.copy()
        if "full_tool_outputs" in checkpoint_copy.get("channel_values", {}):
            del checkpoint_copy["channel_values"]["full_tool_outputs"]
        
        super().put(config, checkpoint_copy, metadata)

    def get_tuple(self, config: dict) -> Optional[Tuple]:
        # Get the saved checkpoint from Redis.
        saved_tuple = super().get_tuple(config)
        if saved_tuple:
            checkpoint = saved_tuple.checkpoint
            # Add back the empty field to conform to the AgentState schema upon loading.
            if "full_tool_outputs" not in checkpoint.get("channel_values", {}):
                checkpoint["channel_values"]["full_tool_outputs"] = []
            return saved_tuple
        return None

def get_redis_checkpointer() -> RedisSaver:
    """
    Factory function to create and configure the RedisSaver from `langgraph-checkpoint-redis`.
    This function provides a Redis-backed checkpointer for LangGraph agents.
    """
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    return RedisSaver(redis_url)
