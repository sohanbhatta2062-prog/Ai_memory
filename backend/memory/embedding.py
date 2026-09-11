from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_ROUTER_KEY = os.getenv("OPENAI_ROUTER_KEY")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENAI_ROUTER_KEY,
)

model = "liquid/lfm-2.5-embedding-350m:free"


def embed_text(text: str):

    try:

        response = client.embeddings.create(
            model=model,
            input=text,
            encoding_format="float"
        )

        embeddings = response.data[0].embedding
    except Exception as e:
        raise ValueError(f"Something went wrong in embed_text : {e}")
    
    return embeddings


# text = "My name is sohan bhatta"

# res = embed_text(text=text)

# print(res[:5])