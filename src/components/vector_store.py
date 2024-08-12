"""
This module provides the VectorStore class for managing a vector database using Qdrant.
"""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, Filter, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from src.logger import logging


class VectorStore:
    """
    A class to manage a vector database using Qdrant.
    """

    def __init__(self, collection_name="greenai_docs"):
        """
        Initialize the VectorStore with a collection name.

        Args:
            collection_name (str): The name of the collection to use.
        """
        logging.info("Initializing VectorStore")

        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.client = QdrantClient("localhost", port=6333)
        self.collection_name = collection_name

        # Create collection if it doesn't exist
        self.create_collection()
        logging.info(f"VectorStore initialized with collection: {self.collection_name}")


    def create_collection(self):
        """
        Create a collection in Qdrant if it doesn't already exist.
        """
        logging.info("Creating collection if it doesn't exist")
        collections = self.client.get_collections()
        if self.collection_name not in [c.name for c in collections.collections]:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )
            logging.info(f"Collection {self.collection_name} created")
        else:
            logging.info(f"Collection {self.collection_name} already exists")

    def add_documents(self, chunks):
        """
        Add documents to the collection.

        Args:
            chunks (list): A list of document chunks to add.
        """
        logging.info("Adding documents to the collection")
        collection_info = self.client.get_collection(self.collection_name)
        start_id = collection_info.points_count

        points = [
            PointStruct(
                id=start_id + i,
                vector=self.encoder.encode(chunk["content"]).tolist(),
                payload={"content": chunk["content"], "source": chunk["source"]},
            )
            for i, chunk in enumerate(chunks)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)
        logging.info(f"Added {len(chunks)} documents to the collection")

    def rewrite_database(self, chunks):
        """
        Rewrite the entire database with new documents.

        Args:
            chunks (list): A list of document chunks to add.
        """
        logging.info("Rewriting the entire database")
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(must=[]),
        )
        logging.info("Deleted all documents from the collection")
        self.add_documents(chunks)
        logging.info("Rewrote the entire database")

    def retrieve(self, query, top_k=3):
        """
        Retrieve the top-k most similar documents to the query.

        Args:
            query (str): The query string.
            top_k (int): The number of top results to return.

        Returns:
            list: A list of the top-k most similar documents.
        """
        logging.info(f"Retrieving top {top_k} documents for query: {query}")
        query_vector = self.encoder.encode(query).tolist()
        results = self.client.search(
            collection_name=self.collection_name, query_vector=query_vector, limit=top_k
        )
        logging.info(f"Retrieved {len(results)} documents")
        return [hit.payload for hit in results]
