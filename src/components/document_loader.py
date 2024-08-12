"""
This module contains the `DocumentLoader` class for loading and chunking documents.
The `DocumentLoader` class has a static method `load_and_chunk_documents`.
This method takes a directory path, chunk size, and overlap as input.
It loads all text files in the directory, reads their content, and chunks it.
Chunks are created based on the given chunk size and overlap.
Chunks are stored in a list of dictionaries, each containing the chunk content and source filename.
"""

import os

import nltk
from nltk.tokenize import sent_tokenize
from src.logger import logging

nltk.download("punkt", quiet=True)


class DocumentLoader:  # pylint: disable=too-few-public-methods
    """
    A class for loading and chunking documents from a directory.
    Methods:
    - load_and_chunk_documents(directory: str, chunk_size: int = 1000, overlap: int = 1)
        Loads and chunks documents from the specified directory.
        Args:
            directory (str): The directory path where the documents are located.
            chunk_size (int, optional): The maximum size of each chunk in characters.
                Defaults to 1000.
            overlap (int, optional): The number of sentences to overlap between chunks.
                Defaults to 1.
        Returns:
            List[Dict[str, str]]: A list of dictionaries,
                where each dictionary represents a chunk of a document.
                Each dictionary contains the following keys:
                - "content" (str): The content of the chunk.
                - "source" (str): The filename of the source document.
    """

    @staticmethod
    def load_and_chunk_documents(
        directory: str, chunk_size: int = 1000, overlap: int = 1
    ):
        """
        Load and chunk documents from the specified directory.
        Args:
            directory (str): The directory path where the documents are located.
            chunk_size (int, optional): The maximum size of each chunk in characters.
                Defaults to 1000.
            overlap (int, optional): The number of sentences to overlap between chunks.
                Defaults to 1.
        Returns:
            List[Dict[str, str]]: A list of dictionaries,
                where each dictionary represents a chunk of content
            with the following keys:
                - "content": The chunk content as a string.
                - "source": The filename of the source document.
        """
        logging.info(f"Loading and chunking documents from directory: {directory}")
        chunks = []
        for filename in os.listdir(directory):
            if filename.endswith(".txt"):
                with open(
                    os.path.join(directory, filename), "r", encoding="utf-8"
                ) as file:
                    content = file.read()
                    sentences = sent_tokenize(content)
                    current_chunk = []
                    current_length = 0

                    for _, sentence in enumerate(sentences):
                        current_chunk.append(sentence)
                        current_length += len(sentence)

                        if current_length >= chunk_size:
                            chunk_content = " ".join(current_chunk)
                            chunks.append(
                                {"content": chunk_content, "source": filename}
                            )

                            overlap_start = max(0, len(current_chunk) - overlap)
                            current_chunk = current_chunk[overlap_start:]
                            current_length = sum(len(s) for s in current_chunk)

                    if current_chunk:
                        chunk_content = " ".join(current_chunk)
                        chunks.append({"content": chunk_content, "source": filename})
        logging.info(f"Loaded and chunked {len(chunks)} documents")
        return chunks
