from chromadb import Documents, EmbeddingFunction, Embeddings
from .embedding_config import get_embedding_model

class LangchainEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        self._embedding_model = get_embedding_model()

    def __call__(self, input: Documents) -> Embeddings:
        return self._embedding_model.embed_documents(input)

def get_chroma_embedding_function() -> LangchainEmbeddingFunction:
    return LangchainEmbeddingFunction()
