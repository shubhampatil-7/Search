from sentence_transformers import SentenceTransformer

class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # self.model.encode(text)


    def verify(self):
        SemanticSearch()

        print(f"Model loaded: {self.model}")
        print(f"Max sequence length: {self.model.max_seq_length}")
