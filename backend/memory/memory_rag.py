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


THRESHOLD = 1.7 # temproray threshold

def recall(query: str, k = 3):

    query_vector = embed_text(text=query)

    results = memory_store._collection.query(
        query_embeddings=[query_vector],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )
    ouputs = []

    for key, text, distance in zip(results["ids"][0], results["documents"][0], results["distances"][0]):

        if distance <= THRESHOLD:

            ouputs.append(
                {
                    "key": key,
                    "text": text,
                    "distance": distance
                }
            )

    # print(ouputs)
    # print("Collection count:", memory_store._collection.count())
    # print("Raw results:", results)
    return ouputs

# key = "current_project"

# text = "Right now i am making a project where a llm model can remeber the conversation history and know me better.."

# add_memory(key=key, text=text)

# query = "Mount everst"

# recall(query=query)

