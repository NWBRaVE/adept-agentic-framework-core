from fastmcp import FastMCP, Context
from pydantic import BaseModel, Field
from typing import Optional

# Import the factory
from ...core.mcp_tool_factory import mcp_tool_factory
from ...core.logger_config import get_logger

logger = get_logger(__name__)

# 1. Define the Pydantic schema for the tool's arguments
class AnalyzeTextInput(BaseModel):
    text: str = Field(description="The text content to analyze.")
    sentiment_analysis: Optional[bool] = Field(
        default=False,
        description="Whether to perform sentiment analysis on the text."
    )

# 2. Define the asynchronous function that contains the tool's core logic
#    This function will receive validated arguments directly.
async def _analyze_text_logic(ctx: Context, args: AnalyzeTextInput): # Changed signature
    """
    Analyzes the provided text and optionally performs sentiment analysis.
    """
    logger.info(f"Analyzing text: {args.text[:50]}...") # Access via args.text
    result = {"original_text_length": len(args.text)}

    if args.sentiment_analysis: # Access via args.sentiment_analysis
        # Placeholder for actual sentiment analysis logic
        logger.info("Performing sentiment analysis...")
        if "happy" in args.text.lower() or "good" in args.text.lower():
            sentiment = "positive"
        elif "sad" in args.text.lower() or "bad" in args.text.lower():
            sentiment = "negative"
        else:
            sentiment = "neutral"
        result["sentiment"] = sentiment
    
    return {"status": "success", "analysis_result": result}

# 3. Create a registration function for this tool
def register_text_analysis_tool(mcp: FastMCP):
    """
    Registers the analyze_text tool with the FastMCP instance.
    """
    # Use the factory to create and apply the decorator to your logic function
    mcp_tool_factory(
        mcp_instance=mcp,
        name="analyze_text",
        description="Analyzes text content, optionally including sentiment analysis."
    )(_analyze_text_logic) # Apply the decorator to the logic function
