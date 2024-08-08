import os
from src.utility import get_root
from nltk.tokenize import sent_tokenize
import nltk

nltk.download('punkt', quiet=True)

class DocumentLoader:
    @staticmethod
    def load_and_chunk_documents(directory: str, chunk_size: int = 1000, overlap: int = 1):
        chunks = []
        for filename in os.listdir(directory):
            if filename.endswith(".txt"):
                with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                    content = file.read()
                    sentences = sent_tokenize(content)
                    current_chunk = []
                    current_length = 0
                    
                    for i, sentence in enumerate(sentences):
                        current_chunk.append(sentence)
                        current_length += len(sentence)
                        
                        if current_length >= chunk_size:
                            chunk_content = " ".join(current_chunk)
                            chunks.append({"content": chunk_content, "source": filename})
                            
                            # Start the next chunk with the overlap
                            overlap_start = max(0, len(current_chunk) - overlap)
                            current_chunk = current_chunk[overlap_start:]
                            current_length = sum(len(s) for s in current_chunk)
                    
                    # Add any remaining content as the last chunk
                    if current_chunk:
                        chunk_content = " ".join(current_chunk)
                        chunks.append({"content": chunk_content, "source": filename})
        
        return chunks

if __name__ == "__main__":
    directory = os.path.join(get_root(), "data", "txt")
    chunks = DocumentLoader.load_and_chunk_documents(directory, chunk_size=1000, overlap=1)
    for i, chunk in enumerate(chunks[:5]):
        print(f"Chunk {i + 1}:")
        print(f"Source: {chunk['source']}")
        print(f"Content: {chunk['content']}")
        print()
        