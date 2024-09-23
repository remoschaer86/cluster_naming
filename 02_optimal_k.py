from dotenv import load_dotenv
load_dotenv()
from src.db import EmbeddingDatabase
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

DB_PATH = "./data/cudosgpt_prompts.db"
MAX_K = 50


with EmbeddingDatabase(DB_PATH) as db:

    # Load the last successful index from the database
    last_item_idx = db.get_last_item_idx()
    
    embeddings = db.get_vectors()

    # Set the maximum number of clusters to be one less than the number of texts
    max_clusters = min(MAX_K, len(embeddings) - 1)

    wss = []  # Within-cluster sum of squares
    silhouette_scores = []

    # Try different numbers of clusters from 1 to max_clusters
    for n_clusters in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embeddings)
        wss.append(kmeans.inertia_)  # Inertia: WSS for KMeans
        
        # Silhouette score is not defined for n_clusters = 1
        if n_clusters > 1:
            score = silhouette_score(embeddings, kmeans.labels_)
            silhouette_scores.append(score)

    # Plot the WSS (Elbow Method)
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, max_clusters + 1), wss, marker='o', linestyle='--')
    plt.title('Elbow Method For Optimal K')
    plt.xlabel('Number of clusters')
    plt.ylabel('Within-cluster Sum of Squares')
    plt.show()

    # Plot Silhouette Scores to support cluster decision
    if silhouette_scores:
        plt.figure(figsize=(10, 5))
        plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o', linestyle='--', color='r')
        plt.title('Silhouette Score vs Number of Clusters')
        plt.xlabel('Number of clusters')
        plt.ylabel('Silhouette Score')
        plt.show()
    else:
        print("Silhouette scores not calculated, fewer than 2 clusters.")
    
