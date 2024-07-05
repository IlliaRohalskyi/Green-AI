from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter
class VectorStore:
    def __init__(self, collection_name="greenai_docs"):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.client = QdrantClient("localhost", port=6333)
        self.collection_name = collection_name
        
        # Create collection if it doesn't exist
        self.create_collection()

    def create_collection(self):
        collections = self.client.get_collections()
        if self.collection_name not in [c.name for c in collections.collections]:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )

    def add_documents(self, chunks):
        # Get the current count of points in the collection
        collection_info = self.client.get_collection(self.collection_name)
        start_id = collection_info.points_count

        # Insert new documents
        points = [
            PointStruct(
                id=start_id + i,
                vector=self.encoder.encode(chunk["content"]).tolist(),
                payload={"content": chunk["content"], "source": chunk["source"]}
            ) for i, chunk in enumerate(chunks)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)

    def rewrite_database(self, chunks):
        # Delete all existing points
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(must=[]),  # This deletes all points
        )

        # Insert new documents
        self.add_documents(chunks)

    def retrieve(self, query, top_k=3):
        query_vector = self.encoder.encode(query).tolist()
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k
        )
        return [hit.payload for hit in results]

if __name__ == "__main__":
    import os
    from src.utility import get_root
    from src.components.document_loader import DocumentLoader

    directory = os.path.join(get_root(), "data", "txt")
    chunks = DocumentLoader.load_and_chunk_documents(directory)

    vector_store = VectorStore()

    # Rewrite the entire database
    #vector_store.rewrite_database(chunks)

    query = "Some query about Green AI"
    results = vector_store.retrieve(query)
    print(results)