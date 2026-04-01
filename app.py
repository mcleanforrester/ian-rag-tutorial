import bootstrap  # noqa: F401 — must run before any LangChain imports

import os
import warnings

warnings.filterwarnings("ignore", message="Could not obtain an event loop")

from langchain.chat_models import init_chat_model
from langchain_ollama import OllamaEmbeddings
from langchain.agents import create_agent

import config
from identity.models import User, INTERNAL
from ingestion.loader import load_department_docs
from ingestion.splitter import split_documents
from retrieval.store import init_vector_store, index_documents
from retrieval.repository import DocumentRepository
from conversation.middleware import create_prompt_middleware
from conversation.repl import run

# 1. Simulated user
user = User(name="Alice", department="engineering", permission_level=INTERNAL)

# 2. LLM
model = init_chat_model(config.MODEL_NAME, model_provider=config.MODEL_PROVIDER)

# 3. Embeddings
embeddings = OllamaEmbeddings(model=config.EMBEDDING_MODEL)

# 4. Vector store
base_dir = os.path.dirname(__file__)
vector_store = init_vector_store(
    persist_directory=os.path.join(base_dir, config.CHROMA_PATH),
    collection_name=config.CHROMA_COLLECTION,
    embedding_function=embeddings,
)

# 5. Ingest documents if store is empty
if vector_store._collection.count() == 0:
    print("No existing Chroma data found. Building index from PDFs...")
    docs = load_department_docs(os.path.join(base_dir, config.PDF_PATH))
    chunks = split_documents(docs, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    print(f"Total splits across all departments: {len(chunks)}")
    index_documents(vector_store, chunks)
else:
    print(f"Loaded existing Chroma index ({vector_store._collection.count()} documents).")

# 6. Repository
repository = DocumentRepository(vector_store)

# 7. Agent
prompt_middleware = create_prompt_middleware(repository, user)
agent = create_agent(model, tools=[], middleware=[prompt_middleware])

# 9. Run
run(agent, user)
