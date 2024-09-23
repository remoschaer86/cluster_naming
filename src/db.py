
import sqlite3
import numpy as np
from src.embedding_record import EmbeddingRecord


class EmbeddingDatabase:
    def __init__(self, db_name: str) -> None:
        self.db_name = db_name
        self._conn = None
        self._cursor = None

    def __enter__(self):
        # _connect to the SQLite database
        self._conn = sqlite3.connect(self.db_name)
        self._cursor = self._conn.cursor()
        self._create_table()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Commit any changes and close the _connection
        if self._conn:
            self._conn.commit()
            self._conn.close()

    def _create_table(self) -> None:
        # Create a table for storing embeddings if it doesn't exist
        self._cursor.execute('''
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY,
            chat_id TEXT,
            text TEXT,
            embedding BLOB,
            success BOOLEAN,
            error TEXT
        )
        ''')
    
    def insert_embedding(self, chat_id: str, text: str, embedding: list[float], success:bool=True, error: str | None=None) -> None:
        # Insert a new embedding into the database
        embedding_vector = np.array(embedding, dtype=np.float32)

        embedding_blob = embedding_vector.tobytes()
        
        self._cursor.execute('''
        INSERT INTO embeddings (chat_id, text, embedding, success, error) 
        VALUES (?, ?, ?, ?, ?)
        ''', (chat_id, text, embedding_blob, success, error))

    def get_all_embedding_records(self) -> list[EmbeddingRecord]:
        # Fetch all embeddings from the database
        self._cursor.execute('SELECT * FROM embeddings')
        rows = self._cursor.fetchall()
        
        # Convert each embedding blob back to a list of floats
        result = []
        for row in rows:
            id, chat_id, text, embedding_blob, success, error = row
            embedding_array = np.frombuffer(embedding_blob, dtype=np.float32)
            embedding_list = embedding_array.tolist()
            record = EmbeddingRecord(id, chat_id, text, embedding_list, success, error)
            result.append(record)
        return result
    
    def get_vectors(self) -> list[list[float]]:
        # Fetch all embeddings from the database
        self._cursor.execute('SELECT embedding FROM embeddings WHERE success=1')
        
        binary_vectors = self._cursor.fetchall()
        
        vectors = []
        for bin_vector in binary_vectors:
            vector = np.frombuffer(bin_vector[0], dtype=np.float32)
            vectors.append(vector)
        return np.array(vectors, dtype=np.float32)
            
            
    
    
 

    def get_last_item_idx(self) -> int:
            # Load the last successful index from the database
        self._cursor.execute('SELECT MAX(id) FROM embeddings')
        return self._cursor.fetchone()[0]

