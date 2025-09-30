from fastmcp import FastMCP, Context
from ..state_manager import create_session_if_not_exists # Assuming session logic might be needed later
from ...logger_config import get_logger
import uuid # Needed if session ID generation is kept local
from typing import Dict, Any, List
import asyncio
import os

# Selenium imports
from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException # Import TimeoutException
from webdriver_manager.chrome import ChromeDriverManager

# Use the centralized logger
logger = get_logger(__name__)

# --- Session Management Helpers (adapt if session context is needed for websearch) ---
async def ensure_session_initialized_websearch(ctx: Context) -> str:
    """Helper to ensure the session exists in the database."""
    session_id = get_stable_session_id_websearch(ctx)
    client_id = ctx.client_id if hasattr(ctx, 'client_id') else None
    await create_session_if_not_exists(session_id, client_id)
    return session_id

def get_stable_session_id_websearch(ctx: Context) -> str:
    """Retrieves or generates a stable session ID for the websearch tool context."""
    if hasattr(ctx, 'session_id') and ctx.session_id:
        return ctx.session_id
    if ctx.request_id:
        return ctx.request_id
    logger.warning("WebSearch Tool: Could not determine a stable session ID. Generating a new one.")
    return str(uuid.uuid4())

def _run_selenium_search(query: str, num_results: int, search_engine_url: str, wait_timeout: int) -> List[Dict[str, str]]:
    """Synchronous function to perform web search using Selenium."""
    results = []
    chrome_options = ChromeOptions()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox") # Important for running as root in Docker
    chrome_options.add_argument("--disable-dev-shm-usage") # Important for Docker
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("window-size=1920x1080") # Can help with some sites

    driver = None
    try:
        logger.info("Initializing ChromeDriver using webdriver_manager...")
        # Use webdriver-manager to automatically download and manage ChromeDriver
        service = ChromeService(ChromeDriverManager().install())
        logger.info("ChromeDriver service initialized.")
        
        driver = webdriver.Chrome(service=service, options=chrome_options)
        logger.info(f"WebDriver initialized. Navigating to: {search_engine_url}?q={query}")
        driver.get(f"{search_engine_url}?q={query}")
        logger.info(f"Successfully navigated to search page for query: {query}")

        # Wait for at least one search result element to load (example for Google)
        # Adjust selectors if using a different search engine
        logger.info("Waiting for search results container (div.g)...")
        WebDriverWait(driver, wait_timeout).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.g")) # Wait for the first result entry
        )
        logger.info("Search results container found.")

        search_entries = driver.find_elements(By.CSS_SELECTOR, "div.g")[:num_results]
        logger.info(f"Found {len(search_entries)} search entry elements.")

        for entry in search_entries:
            try: # Added i for logging, assuming it was intended
                logger.debug(f"Processing search entry {i+1}...")
                title_element = entry.find_element(By.CSS_SELECTOR, "h3")
                link_element = entry.find_element(By.CSS_SELECTOR, "a")
                # Snippet selector might vary, this is a common one
                snippet_elements = entry.find_elements(By.CSS_SELECTOR, "div[data-sncf='pd']") # Google's snippet class
                                
                title = title_element.text
                link = link_element.get_attribute("href")
                snippet = snippet_elements[0].text if snippet_elements else "N/A"
                logger.debug(f"Entry {search_entries.index(entry)+1}: Title='{title}', Link='{link}', Snippet='{snippet[:30]}...'")

                if title and link:
                    results.append({"title": title, "link": link, "snippet": snippet})
            except Exception as e:
                logger.warning(f"Could not parse search result entry {search_entries.index(entry)+1}: {type(e).__name__} - {e}", exc_info=True)
                continue
    except TimeoutException as te:
        logger.error(f"Selenium search timed out after {wait_timeout} seconds waiting for results for query: '{query}'", exc_info=True)
        raise RuntimeError(f"Selenium timed out after {wait_timeout}s waiting for search results for query '{query}'. Original error: {te}") from te
    except Exception as e:
        logger.error(f"Selenium search failed: {type(e).__name__} - {e}", exc_info=True)
        raise RuntimeError(f"Selenium execution failed: {type(e).__name__} - {e}") from e
    finally:
        if driver:
            logger.info("Quitting WebDriver.")
            driver.quit()
    return results

def register_tools(mcp: FastMCP):
    @mcp.tool()
    async def perform_web_search(query: str, ctx: Context, num_results: int = 5, timeout: int = 90, mcp_session_id: str = None) -> Dict[str, Any]:
        """
        Performs a web search using Google (headless Chrome) and returns the top N results.
        """
        logger.info(f"Web search tool called with query: '{query}', num_results: {num_results}, timeout: {timeout}")
        
        session_id = await ensure_session_initialized_websearch(ctx) # Optional: if session context is needed
        await ctx.info(f"Session {session_id}: Performing web search for query: '{query}'")
        
        if not session_id:
            await ctx.error("Session ID is required for web search.")
            return {"error": "Session ID is required for web search."}
        
        search_engine_url = os.getenv("SEARCH_ENGINE_URL", "https://www.google.com/search")

        try:
            # Run the blocking Selenium code in a separate thread
            search_results = await asyncio.to_thread(_run_selenium_search, query, num_results, search_engine_url, timeout)
            
            if not search_results:
                await ctx.warning(f"No search results found for query: '{query}'")
                return {"query": query, "results": [], "message": "No results found."}
            
            await ctx.info(f"Found {len(search_results)} results for query: '{query}'")
            return {"query": query, "results": search_results}
        except Exception as e:
            logger.error(f"Error during web search for query '{query}': {e}", exc_info=True)
            await ctx.error(f"Web search failed: {e}")
            return {"error": f"Failed to perform web search: {e}"}
            
    logger.info("Web search MCP tool registered.")
