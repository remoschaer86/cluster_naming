from dotenv import load_dotenv
load_dotenv()
from openai import AzureOpenAI
import numpy as np
from sklearn.cluster import KMeans
import os
import json
from src.db import EmbeddingDatabase

ai_svc_client = AzureOpenAI(
    api_key=os.getenv("AzureOpenAi__ApiKey"),  
    api_version=os.getenv("AzureOpenAi__ApiVersion"),
    azure_endpoint=os.getenv("AzureOpenAi__BaseAddress")
)

CHAT_MODEL='gpt-4o'

DB_PATH = "./data/cudosgpt_prompts.db"
K = 7


with EmbeddingDatabase(DB_PATH) as db:

    vectors = db.get_vectors()
    
    records = db.get_all_embedding_records()

    # Cluster texts
    kmeans = KMeans(n_clusters=K, random_state=0).fit(vectors)
    labels = kmeans.labels_

    # Group texts by cluster
    clustered_texts = {}
    for i, label in enumerate(labels):
        label = int(label)
        if label not in clustered_texts:
            clustered_texts[label] = []
        clustered_texts[label].append(records[i].text)
    
        
    
    print(clustered_texts[5][0])
    print(clustered_texts[5][1])
    print(clustered_texts[5][2])
    print(clustered_texts[5][3])
    
    
    
    
    
    # for label, texts in clustered_texts.items():
    #     prompt = "Instructions:\nI provide you with some texts which are semantically similar. I want you to create a single label for all of them. Your response should be in a JSON format.  {\"label\": \"<the single label, which should be the same for all items>\"}\n\n"
    #     for idx, text in enumerate(texts):
    #         prompt += f"Text {idx + 1}:\n{text}\n\n"

    #     response = ai_svc_client.chat.completions.create(
    #         model=CHAT_MODEL,
    #             messages=[
    #                 {"role": "system", "content": "You are a helpful assistant whos task it is to categorize texts"},
    #                 {"role": "user", "content": prompt},      
    #         ],
    #             response_format={ "type": "json_object" }
    #     )
    #     # Parse the response content assuming it's a JSON string
    #     category_json = json.loads(response.choices[0].message.content)
        
    #     # Extract the "label" value from the parsed JSON
    #     category_name = category_json.get("label")
        
    #     print(f"Cluster {label} Category: {category_name}")
