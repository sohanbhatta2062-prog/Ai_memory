from langgraph.graph import StateGraph, START, END
from .schemas import State
from .nodes import memory_retrival, context_builder, agent, memory_decision

graph = StateGraph(State)

graph.add_node("memory_retrival", memory_retrival)
graph.add_node("context_builder", context_builder)
graph.add_node("agent", agent)
graph.add_node("memory_decision", memory_decision)

graph.add_edge(START, "memory_retrival")
graph.add_edge("memory_retrival", "context_builder")
graph.add_edge("context_builder", "agent")
graph.add_edge("agent", "memory_decision")
graph.add_edge("memory_decision", END)

compiled_graph = graph.compile()

