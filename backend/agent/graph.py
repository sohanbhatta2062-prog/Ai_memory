from langgraph.graph import StateGraph, START, END
from .schemas import State
from .nodes import memory_retrival, context_builder

graph = StateGraph(State)

graph.add_node("memory_retrival", memory_retrival)
graph.add_node("context_builder", context_builder)


graph.add_edge(START, "memory_retrival")
graph.add_edge("memory_retrival", "context_builder")

compiled_graph = graph.compile()

