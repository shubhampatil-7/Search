from sentence_transformers import SentenceTransformer
import numpy as np
import os
from search_utils import MOVIE_EMBEDDINGS_PATH, load_movies

class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def verify(self):
        SemanticSearch()

        print(f"Model loaded: {self.model}")
        print(f"Max sequence length: {self.model.max_seq_length}")
    
    
    def generate_embedding(self, text: str):
        if len(text) == 0 or text.isspace():
            raise ValueError("Input text cannot be empty or whitespace.")
        
        encoding = self.model.encode([text])
        return encoding[0]

    def build_embeddings(self, documents):
        self.documents = documents
        self.document_map = {}

        stringlist: list[str] = []
        for doc in documents:
            self.document_map[doc["id"]] = doc

            stringlist.append(f"{doc['title']}: {doc['description']}")
            
        self.embeddings = self.model.encode(stringlist, show_progress_bar=True)
        os.makedirs(os.path.dirname(MOVIE_EMBEDDINGS_PATH), exist_ok=True)
        np.save(MOVIE_EMBEDDINGS_PATH, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        self.document_map = {}
        for doc in documents:
            self.document_map[doc["id"]] = doc

        if os.path.exists(MOVIE_EMBEDDINGS_PATH):
            self.embeddings = np.load(MOVIE_EMBEDDINGS_PATH)
            if len(self.embeddings) == len(self.documents):
                return self.embeddings
            
        return self.build_embeddings(documents)
    

def embed_query_text(query):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")
    


def verify_embeddings():
    ss = SemanticSearch()
    documents = load_movies()
    embeddings = ss.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )
        


def embed_text(text: str):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")
