from agent.nodes import memory_retrival,context_builder
from agent.schemas import State
from agent.llm_models import google_llm, openai_llm
from agent.graph import compiled_graph

query = input("Enter your query: ")

state = State(
    messages=[],
    query=query,
    search_result = "",
    retrieved_memories=[],
    context="",
    final_result="",
    should_save=False
)


res = compiled_graph.invoke(state)

print(f"\n\n{res}")

print(f"\n\n{res["final_result"]}")

print(f"\n\n{res["should_save"]}")



# res = google_llm.invoke("Just say ok")

# print(res)
