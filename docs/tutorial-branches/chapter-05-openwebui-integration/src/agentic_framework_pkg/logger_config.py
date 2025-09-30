import logging
import os
import sys
import json # Using json.dumps for args for better readability

# Attempt to import Langchain message types. If not available, the formatter will still work but won't pretty-print them.
try:
    from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage, ToolMessage
    LANGCHAIN_MESSAGES_AVAILABLE = True
except ImportError:
    LANGCHAIN_MESSAGES_AVAILABLE = False
    # Define dummy classes if langchain_core is not available, so the formatter doesn't break
    class BaseMessage: pass
    class AIMessage(BaseMessage): pass
    class HumanMessage(BaseMessage): pass
    class SystemMessage(BaseMessage): pass
    class ToolMessage(BaseMessage): pass

DEFAULT_LOG_LEVEL_STR = "INFO" # Changed to string for os.environ.get
LOG_LEVEL_STR = os.environ.get("LOGGING_LEVEL", DEFAULT_LOG_LEVEL_STR).upper()
LOG_LEVEL = getattr(logging, LOG_LEVEL_STR, logging.INFO) # Convert string to logging level

DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

class ReadableLangchainMessageFormatter(logging.Formatter):
    """
    A custom logging.Formatter to produce more readable output for Langchain messages.
    """
    def _format_message(self, message: BaseMessage) -> str:
        if not LANGCHAIN_MESSAGES_AVAILABLE:
            return str(message)

        if isinstance(message, AIMessage):
            tool_calls_summary = []
            if message.tool_calls:
                for tc in message.tool_calls:
                    args_str = json.dumps(tc.get('args', {}))
                    tool_calls_summary.append(
                        f"ToolCall(name='{tc.get('name')}', args={args_str}, id='{tc.get('id')}')"
                    )
            usage_summary_parts = []
            if message.usage_metadata:
                if message.usage_metadata.get('input_tokens') is not None:
                    usage_summary_parts.append(f"in:{message.usage_metadata['input_tokens']}")
                if message.usage_metadata.get('output_tokens') is not None:
                    usage_summary_parts.append(f"out:{message.usage_metadata['output_tokens']}")
                if message.usage_metadata.get('total_tokens') is not None:
                    usage_summary_parts.append(f"total:{message.usage_metadata['total_tokens']}")
            usage_summary = f", Usage({', '.join(usage_summary_parts)})" if usage_summary_parts else ""
            content_display = message.content[:100] + "..." if len(message.content) > 100 else message.content
            finish_reason = message.response_metadata.get('finish_reason', 'N/A') if message.response_metadata else 'N/A'
            return (
                f"AIMessage(id='{message.id}', content='{content_display}', "
                f"finish_reason='{finish_reason}', "
                f"tool_calls=[{'; '.join(tool_calls_summary)}]{usage_summary})"
            )
        elif isinstance(message, (HumanMessage, SystemMessage, ToolMessage)): # Simplified for brevity
            content_display = message.content[:100] + "..." if len(message.content) > 100 else message.content
            return f"{type(message).__name__}(id='{message.id}', content='{content_display}')"
        return str(message)

    def format(self, record: logging.LogRecord) -> str:
        if LANGCHAIN_MESSAGES_AVAILABLE:
            if isinstance(record.msg, BaseMessage):
                record.msg = self._format_message(record.msg)
            if record.args and isinstance(record.args, tuple):
                new_args = [self._format_message(arg) if isinstance(arg, BaseMessage) else arg for arg in record.args]
                record.args = tuple(new_args)
        return super().format(record)

def get_logger(logger_name: str, level: int = LOG_LEVEL) -> logging.Logger:
    """
    Creates and configures a logger instance.

    Args:
        logger_name: The name for the logger (typically __name__).
        level: The logging level.

    Returns:
        A configured logging.Logger instance.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    # Add console handler if no handlers are configured
    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        # Use the custom formatter
        formatter = ReadableLangchainMessageFormatter(DEFAULT_LOG_FORMAT)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger