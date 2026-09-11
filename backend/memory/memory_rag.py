import chromadb
from langchain_chroma import Chroma
from datetime import datetime
from .embedding import embed_text

client = chromadb.PersistentClient(path="./chroma_data")

memory_store = Chroma(
    client=client,
    collection_name="memory"
)

def add_memory(key: str, text: str, metadata: dict = None):
    """
    key = a stable identifier for *what* this fact is about, e.g. 'theme_preference'

    """
    if not key or not text:
        raise ValueError(f"Key and text both are neccessary for adding memory!")

    metadata = metadata or {}
    metadata.setdefault("timestamp", datetime.now().isoformat())
    metadata["key"] = key

    vector = embed_text(text=text)

    memory_store._collection.upsert(
        ids=[key],
        documents= [text],
        metadatas=[metadata],
        embeddings=[vector],
    )

    print("Successfully added to memory!!!")


def recall(query: str, k = 2):

    query_vector = embed_text(text=query)

    results = memory_store._collection.query(
        query_embeddings=query_vector,
        n_results=k
    )
    ouputs = []

    for key, text in zip(results["ids"][0], results["documents"][0]):
        ouputs.append(
            {
                "key": key,
                "text": text
            }
        )

    # print(ouputs)

# key = "self_info"

# text = "My name is Sohan Bhatta."

# # add_memory(key=key, text=text)

# query = "What is my name?"

# recall(query=query)