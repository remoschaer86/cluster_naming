from dotenv import load_dotenv
load_dotenv()
import pandas as pd
import numpy as np
import os
import sqlite3
from openai import AzureOpenAI
import time
from src.db import EmbeddingDatabase

# Initialize Azure OpenAI client
ai_svc_client = AzureOpenAI(
    api_key=os.getenv("AzureOpenAi__ApiKey"),  
    api_version=os.getenv("AzureOpenAi__ApiVersion"),
    azure_endpoint=os.getenv("AzureOpenAi__BaseAddress")
)

EMBEDDING_MODEL = "text-embedding-ada-002"
DB_PATH = "./data/cudosgpt_prompts.db"

# Load the DataFrame
df = pd.read_excel('./data/TB01_20240921_224743.xlsx')

# Clean the DataFrame
df_cleaned = df.dropna(how='all').dropna(subset=['Title'])

with EmbeddingDatabase(DB_PATH) as db:

    # Load the last successful index from the database
    last_item_idx = db.get_last_item_idx()
    start_index = last_item_idx + 1 if last_item_idx is not None else 0

    # Process and store embeddings in the database
    for idx in range(start_index, len(df_cleaned)):
        text = df_cleaned['Title'].iloc[idx]
        chat_id = df_cleaned['%chat'].iloc[idx]
        
        try:
            ai_svc_response = ai_svc_client.embeddings.create(input=text, model=EMBEDDING_MODEL)        
            db.insert_embedding(chat_id=chat_id, text=text, embedding=ai_svc_response.data[0].embedding)
            
            print(f"inserting idx: {idx}")

        except Exception as e:
            print(f"Error at index {idx}: {e}")
            db.insert_embedding(chat_id=chat_id, text=text, embedding=None, success=False, error=str(e))
            time.sleep(15)
            
