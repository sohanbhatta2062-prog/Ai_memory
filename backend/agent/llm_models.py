from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os
from .schemas import MemoryDecision

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_ROUTER_KEY = os.getenv("OPENAI_ROUTER_KEY")


google_llm = init_chat_model(
    model="gemini-2.5-flash",
    max_output_tokens=2048,
    thinking_budget=256,
    model_provider="google_genai",
    api_key=GOOGLE_API_KEY,
    temperature=0
)

openai_llm = init_chat_model(
    model="qwen/qwen3-32b",
    model_provider="openai",
    api_key=OPENAI_ROUTER_KEY,
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)

extraction_llm = init_chat_model(
    model="gemini-2.5-flash",
    model_provider="google_genai",
    max_output_tokens=512,
    thinking_budget=0,
)


ollama_llm = init_chat_model(
    model="qwen3:4b",
    model_provider="ollama",
    temperature=0.7,
)


decision_llm = extraction_llm.with_structured_output(MemoryDecision)

