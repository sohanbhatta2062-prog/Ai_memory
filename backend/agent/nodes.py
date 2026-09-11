from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from memory.embedding import embed_text
from memory.memory_rag import add_memory, recall
from .llm_models import openai_llm, google_llm
from .schemas import State


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


def context_builder(state: State) -> dict:

    query = state.query
    recent_messages = state.messages[-6:]
    retrieved_memory = state.retrieved_memories

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a context builder for an AI assistant.

                Your job is to create a clear and relevant context for the
                final answering LLM.

                Use the provided conversation and memories to create useful
                context.

                Do not answer the user's query.
                Return only the useful context.
                """
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

    chain = prompt | google_llm

    try:

        res = chain.invoke(
            {
                "query": query,
                "recent_messages": recent_messages,
                "retrieved_memories": retrieved_memory
            }
        )

        print(res.content)
        return {
            "context": res.content
        }

    except Exception as e:
        raise ValueError(
            f"Something went wrong with context builder node: {e}"
        )

    

