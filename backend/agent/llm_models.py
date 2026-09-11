from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_ROUTER_KEY = os.getenv("OPENAI_ROUTER_KEY")


google_llm = init_chat_model(
    model="gemini-2.5-flash",
    model_provider="google_genai",
    api_key=GOOGLE_API_KEY,
    temperature=0
)

openai_llm = init_chat_model(
    model="minimax/minimax-m3:free",
    model_provider="openai",
    api_key=OPENAI_ROUTER_KEY,
    temperature=0,
    base_url = "https://openrouter.ai/api/v1"
)

