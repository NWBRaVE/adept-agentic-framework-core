import pytest
import os
from unittest.mock import patch, AsyncMock, MagicMock
from agentic_framework_pkg.core.llm_agnostic_layer import LLMAgnosticClient, LLMServiceError

# Mock environment variables for consistent testing
@pytest.fixture(autouse=True)
def mock_env_vars():
    with patch.dict(os.environ, {
        "OPENAI_API_KEY": "test_openai_key",
        "AZURE_API_KEY": "test_azure_key",
        "AZURE_API_BASE": "https://test.openai.azure.com/",
        "AZURE_API_VERSION": "2024-02-01",
        "NVIDIA_API_KEY": "test_nvidia_key",
        "NVIDIA_MULTI_MODAL_MODEL_NAME": "test_nvidia_model",
        "DEFAULT_LLM_MODEL": "test_default_model",
        "EMBEDDING_DEFAULT_MODEL": "test_embedding_model",
        "RAG_DEFAULT_MODEL": "test_rag_model",
        "STREAMLIT_DEFAULT_MODEL": "test_streamlit_model",
        "LITELLM_VERBOSE": "False",
        "USE_SPLIT_STREAM_GRAPH": "false",
        "USE_CHECKPOINTING": "false",
    }, clear=True): # clear=True ensures a clean slate for each test
        yield

# Mock external API client constructors and internal methods
@pytest.fixture(autouse=True)
def mock_llm_agnostic_client_internals():
    with (
        patch("openai.AsyncAzureOpenAI", new_callable=MagicMock) as mock_azure_openai_client_constructor,
        patch("openai.AsyncOpenAI", new_callable=MagicMock) as mock_openai_client_constructor,
        patch("agentic_framework_pkg.core.llm_agnostic_layer.LLMAgnosticClient._call_azure_sdk_acompletion", new_callable=AsyncMock) as mock_azure_acompletion,
        patch("agentic_framework_pkg.core.llm_agnostic_layer.LLMAgnosticClient._call_azure_sdk_aembedding", new_callable=AsyncMock) as mock_azure_aembedding,
        patch("agentic_framework_pkg.core.llm_agnostic_layer.LLMAgnosticClient._initialize_nvidia_client", return_value=None) as mock_nvidia_init,
    ):
        # Configure mocks for return values
        mock_azure_acompletion.return_value = MagicMock(choices=[MagicMock(message=MagicMock(content="Azure SDK completion mock"))])
        mock_azure_aembedding.return_value = [MagicMock(embedding=[0.1, 0.2, 0.3])]

        yield mock_azure_openai_client_constructor, mock_openai_client_constructor, mock_azure_acompletion, mock_azure_aembedding, mock_nvidia_init

@pytest.fixture
def client():
    return LLMAgnosticClient()

@pytest.mark.asyncio
@patch("litellm.acompletion", new_callable=AsyncMock)
async def test_agenerate_response_openai(mock_acompletion, client, mock_llm_agnostic_client_internals):
    # Ensure Azure env vars are NOT set for this test to force OpenAI path
    os.environ.pop("AZURE_API_KEY", None)
    os.environ.pop("AZURE_API_BASE", None)
    os.environ.pop("AZURE_API_VERSION", None)

    mock_acompletion.return_value = MagicMock(choices=[MagicMock(message=MagicMock(content="Test response"))])
    response = await client.agenerate_response(messages=[{"role": "user", "content": "Hello"}], llm_purpose="agent_main")
    assert response.choices[0].message.content == "Test response"
    mock_acompletion.assert_called_once()

@pytest.mark.asyncio
@patch("litellm.acompletion", new_callable=AsyncMock)
async def test_agenerate_response_azure(mock_acompletion, client, mock_llm_agnostic_client_internals):
    # Ensure Azure env vars are set for this test
    os.environ["AZURE_API_KEY"] = "test_azure_key"
    os.environ["AZURE_API_BASE"] = "https://test.openai.azure.com/"
    os.environ["AZURE_API_VERSION"] = "2024-02-01"
    
    mock_acompletion.return_value = MagicMock(choices=[MagicMock(message=MagicMock(content="Azure response"))])
    response = await client.agenerate_response(messages=[{"role": "user", "content": "Hello"}], llm_purpose="agent_main", model="azure/test_deployment")
    assert response.choices[0].message.content == "Azure SDK completion mock" # Corrected assertion
    # Verify that the Azure SDK path was taken, not litellm.acompletion directly
    mock_llm_agnostic_client_internals[2].assert_called_once() # mock_azure_acompletion
    mock_acompletion.assert_not_called()

@pytest.mark.asyncio
@patch("litellm.aembedding", new_callable=AsyncMock)
async def test_acreate_embedding(mock_aembedding, client, mock_llm_agnostic_client_internals):
    # Ensure Azure env vars are set for this test
    os.environ["AZURE_API_KEY"] = "test_azure_key"
    os.environ["AZURE_API_BASE"] = "https://test.openai.azure.com/"
    os.environ["AZURE_API_VERSION"] = "2024-02-01"

    mock_aembedding.return_value = MagicMock(data=[MagicMock(embedding=[0.1, 0.2, 0.3])])
    embeddings = await client.acreate_embedding(input_texts=["Test text"])
    assert embeddings[0].embedding == [0.1, 0.2, 0.3]
    # Verify that the Azure SDK path was taken, not litellm.aembedding directly
    mock_llm_agnostic_client_internals[3].assert_called_once() # mock_azure_aembedding
    mock_aembedding.assert_not_called()

@pytest.mark.asyncio
@patch("openai.AsyncOpenAI.chat.completions.create", new_callable=AsyncMock)
async def test_aextract_text_from_image_content(mock_create, client, mock_llm_agnostic_client_internals):
    # Ensure NVIDIA client is initialized for this test
    client.is_nvidia_client_available = True
    client._nvidia_async_client = MagicMock()
    client._nvidia_async_client.chat.completions.create = mock_create # Assign the mock

    mock_create.return_value = MagicMock(choices=[MagicMock(message=MagicMock(content="Extracted text"))])
    text = await client.aextract_text_from_image_content(image_bytes=b"dummy_image", image_mime_type="image/png", prompt_text="Describe image")
    assert text == "Extracted text"
    mock_create.assert_called_once()

@patch("langchain_openai.AzureChatOpenAI")
@patch("langchain_openai.ChatOpenAI")
@patch("langchain_community.chat_models.ChatLiteLLM")
def test_get_langchain_chat_model(mock_litellm, mock_openai, mock_azure_openai, client, mock_llm_agnostic_client_internals):
    # Test Azure configuration
    os.environ["AZURE_API_KEY"] = "test_azure_key"
    os.environ["AZURE_API_BASE"] = "https://test.openai.azure.com/"
    os.environ["AZURE_API_VERSION"] = "2024-02-01"
    model_instance = client.get_langchain_chat_model(llm_purpose="agent_main", model_name="azure_deployment")
    mock_azure_openai.assert_called_once_with(
        azure_deployment="azure_deployment",
        openai_api_version="2024-02-01",
        azure_endpoint="https://test.openai.azure.com/",
        api_key="test_azure_key",
    )
    assert isinstance(model_instance, MagicMock) # Check if it returns the mocked instance
    mock_azure_openai.reset_mock() # Reset mock for next test

    # Test OpenAI configuration
    os.environ.pop("AZURE_API_KEY", None)
    os.environ.pop("AZURE_API_BASE", None)
    os.environ.pop("AZURE_API_VERSION", None)
    os.environ["OPENAI_API_KEY"] = "test_openai_key"
    model_instance = client.get_langchain_chat_model(llm_purpose="agent_main", model_name="gpt-4")
    mock_openai.assert_called_once_with(
        model="gpt-4",
        openai_api_key="test_openai_key",
    )
    assert isinstance(model_instance, MagicMock)
    mock_openai.reset_mock()

    # Test LiteLLM fallback
    os.environ.pop("OPENAI_API_KEY", None)
    model_instance = client.get_langchain_chat_model(llm_purpose="agent_main", model_name="ollama/llama2")
    mock_litellm.assert_called_once_with(model="ollama/llama2")
    assert isinstance(model_instance, MagicMock)
    mock_litellm.reset_mock()

    # Test ValueError for missing model
    os.environ.pop("OPENAI_API_KEY", None)
    os.environ.pop("AZURE_API_KEY", None)
    os.environ.pop("AZURE_API_BASE", None)
    os.environ.pop("AZURE_API_VERSION", None)
    os.environ.pop("DEFAULT_LLM_MODEL", None)
    os.environ.pop("EMBEDDING_DEFAULT_MODEL", None)
    os.environ.pop("RAG_DEFAULT_MODEL", None)
    os.environ.pop("STREAMLIT_DEFAULT_MODEL", None)
    with pytest.raises(ValueError, match="No model name specified and no default model found"):
        LLMAgnosticClient().get_langchain_chat_model(llm_purpose="non_existent_purpose")
