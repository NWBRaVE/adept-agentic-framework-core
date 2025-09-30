import aiosqlite
import os
import asyncio
import chromadb
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy import text, inspect
from ..core.logger_config import get_logger

logger = get_logger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./data/agentic_framework.db")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")

_db_connection = None
_db_lock = asyncio.Lock()

_chroma_client = None
_chroma_lock = asyncio.Lock()

_async_engine = None
_async_engine_lock = asyncio.Lock()

async def get_db_connection():
    """
    Returns a thread-safe, singleton database connection.
    """
    global _db_connection
    if _db_connection is None:
        async with _db_lock:
            if _db_connection is None:
                db_path = os.path.dirname(DATABASE_URL)
                os.makedirs(db_path, exist_ok=True)
                _db_connection = await aiosqlite.connect(DATABASE_URL)
                logger.info(f"Database connection initialized at {DATABASE_URL}")
    return _db_connection

async def get_chroma_client():
    """
    Returns a thread-safe, singleton ChromaDB client.
    """
    global _chroma_client
    if _chroma_client is None:
        async with _chroma_lock:
            if _chroma_client is None:
                _chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
                logger.info(f"ChromaDB client initialized at {CHROMA_DB_PATH}")
    return _chroma_client

async def get_async_db_engine():
    """
    Returns a thread-safe, singleton SQLAlchemy async engine.
    """
    global _async_engine
    if _async_engine is None:
        async with _async_engine_lock:
            if _async_engine is None:
                _async_engine = create_async_engine(DATABASE_URL, echo=False)
                logger.info(f"SQLAlchemy async engine initialized for URI: {DATABASE_URL}")
    return _async_engine

async def execute_async_sql_query(query: str) -> list[dict]:
    """
    Executes a raw SQL query asynchronously and returns results as a list of dictionaries.
    """
    engine = await get_async_db_engine()
    async with engine.connect() as conn:
        result = await conn.execute(text(query))
        # For SELECT statements, fetch results
        if query.strip().lower().startswith("select"):
            # Convert Row objects to dictionaries
            return [row._asdict() for row in result.fetchall()]
        else:
            # For DDL/DML, just return success status
            return {"status": "success", "message": "Query executed successfully."}

async def get_async_table_info() -> str:
    """
    Retrieves schema information for all tables in the database asynchronously.
    """
    engine = await get_async_db_engine()
    async with engine.connect() as conn:
        # Use run_sync to perform synchronous inspection operations
        # The callable passed to run_sync receives a synchronous connection
        def _get_schema_sync(sync_conn):
            inspector = inspect(sync_conn)
            table_names = inspector.get_table_names()
            
            schema_info = []
            for table_name in table_names:
                columns = inspector.get_columns(table_name)
                column_info = ", ".join([f"{col['name']} {col['type']}" for col in columns])
                schema_info.append(f"Table: {table_name} ({column_info})")
            return "\n".join(schema_info)

        schema = await conn.run_sync(_get_schema_sync)
        return schema

async def initialize_db():
    """
    Initializes the database by creating the singleton connections.
    """
    await get_db_connection()
    await get_chroma_client()
    await get_async_db_engine() # Initialize the async engine

async def close_db_connection():
    """
    Closes the singleton database connection if it exists.
    """
    global _db_connection
    if _db_connection:
        await _db_connection.close()
        _db_connection = None
        logger.info("Database connection closed.")
    
    global _async_engine
    if _async_engine:
        await _async_engine.dispose()
        _async_engine = None
        logger.info("SQLAlchemy async engine disposed.")
