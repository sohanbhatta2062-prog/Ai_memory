from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from memory.embedding import embed_text
from memory.memory_rag import add_memory, recall
from .llm_models import openai_llm, google_llm, ollama_llm, extraction_llm, decision_llm
from .schemas import State
from langchain_core.messages import BaseMessage
from .prompts import CONTEXT_BUILDER_SYSTEM_PROMPT, AGENT_SYSTEM_PROMPT,MEMORY_DICISION_PROMPT

def memory_retrival(state: State) -> dict:
    query = state.query

    if not query:
        raise ValueError("Memory retricval do not have query!")

    try:
        res = recall(query=query)
    except Exception as e:
        raise ValueError(f"Something went wrong with memory retrival node : {e}")

    return {
        "retrieved_memories": res
    }

def _format_conversation(messages: list[BaseMessage]) -> str:
    if not messages:
        return "(no previous messages)"

    role_map = {
        "human": "User",
        "ai": "Assistant",
        "system": "System"
    }

    return "\n".join(
        f"{role_map.get(m.type, m.type)}: {m.content}"
        for m in messages
    )

def _format_memories(memories: list[dict]) -> str:
    if not memories:
        return "(no memories retrieved)"

    lines = [
        f"- {m['text']}"
        for m in memories
        if m.get("text")
    ]

    return "\n".join(lines) if lines else "(no memories retrieved)"

def context_builder(state: State) -> dict:

    query = state.query
    recent_messages = _format_conversation(state.messages[-6:])
    retrieved_memory = _format_memories(state.retrieved_memories)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system", CONTEXT_BUILDER_SYSTEM_PROMPT
            ),
            (
                "human",
                """
                Current Query:
                {query}

                Recent Conversation:
                {recent_messages}

                Retrieved Memories:
                {retrieved_memories}

                Build concise context that will help the final LLM answer
                the current query.
                """
            )
        ]
    )

    chain = prompt | extraction_llm

    try:

        res = chain.invoke(
            {
                "query": query,
                "recent_messages": recent_messages,
                "retrieved_memories": retrieved_memory
            }
        )

    except Exception as e:
        raise ValueError(
            f"Something went wrong with context builder node: {e}"
        )

    content = (res.content or "").strip()
    if content.upper() in ("", "NO_CONTEXT"):
        content = ""
    return {"context": content}

def agent(state: State) -> dict:

    query = state.query
    context = state.context or "No relevant context"

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system", AGENT_SYSTEM_PROMPT

            ),
            (
                "human",
                """
                current query : {query}
                context: {context}
                """

            )
        ]
    )


    chain = prompt | openai_llm

    try:
        res = chain.invoke(
            {
                "query": query,
                "context": context
            }
        )

    except Exception as e:
        raise ValueError(f"Something went wrong in agent node! : {e}")

    return {
        "final_result": res.content,
        "messages": [res]
    }

def memory_decision(state: State) -> dict:

    final_result = state.final_result
    query = state.query

    if not final_result or not query:
        raise ValueError(f"Final answer and query are needed inorder to determine whether to save or not!")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", MEMORY_DICISION_PROMPT),
            (
                "human",
                """
                user's query : {query}
                final_result : {final_result}
                """
            )
        ]
    )

    chain = prompt | decision_llm

    try:
        res = chain.invoke(
            {
                "query": query,
                "final_result": final_result
            }
        )
    except Exception as e:
        raise ValueError(f"Something went wrong in memory decison: {e}")
    print(res)
    print(res.should_save)
    return {
        "should_save": res.should_save
    }

