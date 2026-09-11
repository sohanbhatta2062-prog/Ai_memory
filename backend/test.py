from agent.nodes import memory_retrival,context_builder
from agent.schemas import State
from agent.llm_models import google_llm, openai_llm
from agent.graph import compiled_graph


state = State(
    messages=[],
    query="What is my name?",
    search_result = "",
    retrieved_memories=[],
    context="",
    final_result=""
)


res = compiled_graph.invoke(state)

print(res)

print(f"\n\n{state}")

# res = google_llm.invoke("Just say ok")

# print(res)