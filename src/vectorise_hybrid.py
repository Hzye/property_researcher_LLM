import os
from pathlib import Path
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    VectorParams, 
    PointStruct
)
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain.vectorstores import Qdrant
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

FOLDER_PATH = Path(config["download_dir"])
COLLECTION_NAME = config["collection_name"]
DENSE_MODEL = config["dense_model"]

def load_documents(files):
    """Load text files from directory.

    Args:
        files (list[str]): List of text file names.

    Returns:
        documents (list[str]): List of texts.
    """
    documents = []
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            text = f.read()
            documents.append({"content": text, "metadata": {"source": str(file)}})
    return documents

def chunk_text(documents):
    """_summary_

    Args:
        documents (list[str]): List of texts.

    Returns:
        chunks (list[dict]): List of chunked texts in document format.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = []
    for doc in documents:
        doc_chunks = text_splitter.split_text(doc["content"])
        for chunk in doc_chunks:
            chunks.append({"content": chunk, "metadata": doc["metadata"]})

    return chunks

def compute_sparse_embeddings(texts):
    """Get sparse embeddings.

    Args:
        texts (list[str]): List of texts only.

    Returns:
        sparse_vectors (np.array): Sparse vectors corresponding to input text.
    """
    # compute sparse embeddings (TF-IDF)
    tfidf = TfidfVectorizer()
    sparse_matrix = tfidf.fit_transform(texts)
    sparse_vectors = sparse_matrix.toarray()  # dense numpy array
    
    return sparse_vectors

def compute_dense_embeddings(texts):
    """Get dense embeddings.

    Args:
        texts (list[str]): List of texts only.

    Returns:
        np.array: Dense vectors corresponding to input text.
    """
    dense_model = SentenceTransformer(DENSE_MODEL)
    return dense_model.encoder(texts)

def create_points(texts, chunks, dense_vectors, sparse_vectors):
    """Create vector points.

    Args:
        texts (list[str]): List of texts only.
    """
    points = []
    for i in range(len(texts)):
        point = PointStruct(
            id=i,  # Use integer ID for simplicity and performance
            vector={
                "dense": dense_vectors[i].tolist(),
                "sparse": sparse_vectors[i].tolist()
            },
            payload={
                "text": texts[i],  # Payload should match the chunk text
                "source": chunks[i]["metadata"]["source"] # Add metadata
            }
        )
        points.append(point)
    
    return points

if __name__ == "__main__":
    # get file names
    files = list(FOLDER_PATH.glob("*.txt"))

    # init client
    client = QdrantClient(path="db")

    # load files
    documents = load_documents(files)

    # chunk text
    chunks = chunk_text(documents)
    texts = [chunk["content"] for chunk in chunks]

    sparse_vectors = compute_sparse_embeddings(texts)
    dense_vectors = compute_dense_embeddings(texts)

    VECTOR_SIZE_DENSE = dense_vectors.shape[1]
    VECTOR_SIZE_SPARSE = sparse_vectors.shape[1]

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": VectorParams(size=VECTOR_SIZE_DENSE, distance="Cosine"),
            "sparse": VectorParams(size=VECTOR_SIZE_SPARSE, distance="Dot")
        }
    )

    points = create_points(texts, chunks, dense_vectors, sparse_vectors)

    client.upload_points(
        collection_name=COLLECTION_NAME,
        points=points
    )