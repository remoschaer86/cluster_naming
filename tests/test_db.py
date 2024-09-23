import pytest
import os
import numpy as np
from dotenv import load_dotenv
from openai import AzureOpenAI
from src.db import EmbeddingDatabase
from src.embedding_record import EmbeddingRecord  # Assuming the EmbeddingRecord dataclass is in models.py

load_dotenv()

EMBEDDING_MODEL = "text-embedding-ada-002"
NR_VECTOR_DIMS = 1536

@pytest.fixture
def db_connection():
    # Use a temporary database file for testing
    db = EmbeddingDatabase('test_embeddings.db')
    yield db
    os.remove('test_embeddings.db')  # Clean up after test

@pytest.fixture
def azure_openai_client():
    # Create a real Azure OpenAI client using the environment variables
    return AzureOpenAI(
        api_key=os.getenv("AzureOpenAi__ApiKey"),
        api_version=os.getenv("AzureOpenAi__ApiVersion"),
        azure_endpoint=os.getenv("AzureOpenAi__BaseAddress")
    )

def test_embedding_insertion_and_retrieval(db_connection, azure_openai_client):
    with db_connection as db:
        # Act: Generate embedding from the actual Azure OpenAI service
        ai_svc_response = azure_openai_client.embeddings.create(input="some random text", model=EMBEDDING_MODEL)
        
        # Insert the embedding into the database
        db.insert_embedding(chat_id="some id", text="some text", embedding=ai_svc_response.data[0].embedding)
        
        # Act: Retrieve stored embeddings
        stored_embeddings = db.get_all_embedding_records()

        # Assert: Check that one embedding was stored
        assert len(stored_embeddings) == 1
        
        # Assert: Check that the embedding has the expected length (assuming 1536 for `text-embedding-ada-002`)
        assert len(stored_embeddings[0].embedding) == NR_VECTOR_DIMS
        
        # Assert: Check that the retrieved record is an instance of EmbeddingRecord
        assert isinstance(stored_embeddings[0], EmbeddingRecord)
