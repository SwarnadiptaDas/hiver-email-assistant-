import json
import os
from pathlib import Path
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class EmailRetriever:
    def __init__(self, dataset_path: str = None):
        if dataset_path is None:
            dataset_path = str(Path(__file__).resolve().parent.parent / 'dataset' / 'email_dataset.json')
        self.dataset_path = dataset_path
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.dataset = []
        self.index = None
        self.index_path = str(Path(self.dataset_path).parent / 'email_index.faiss')
        self._load_dataset()
        self.build_index()

    def _load_dataset(self) -> None:
        try:
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.dataset = data.get("emails", [])
        except (FileNotFoundError, json.JSONDecodeError):
            self.dataset = []

    def get_dataset(self) -> list[dict]:
        return self.dataset

    def build_index(self) -> None:
        if not self.dataset:
            return

        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            return

        texts = []
        for email_item in self.dataset:
            incoming = email_item.get("incoming_email", {})
            subject = incoming.get("subject", "")
            body = incoming.get("body", "")
            texts.append(f"{subject}\n{body}")

        if not texts:
            return

        embeddings = self.model.encode(texts, show_progress_bar=False)
        embeddings = np.array(embeddings).astype('float32')

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        
        # Ensure directory exists before caching
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)

    def retrieve(self, email_text: str, top_k: int = 3) -> list[dict]:
        if self.index is None or not self.dataset:
            return []
            
        k = min(top_k, len(self.dataset))
        if k == 0:
            return []

        query_embedding = self.model.encode([email_text], show_progress_bar=False)
        query_embedding = np.array(query_embedding).astype('float32')

        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for idx in indices[0]:
            if idx < len(self.dataset):
                results.append(self.dataset[idx])
                
        return results
