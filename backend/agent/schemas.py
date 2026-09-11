from pydantic import BaseModel
from typing import List, Annotated, Optional
from langgraph.graph import add_messages

class State(BaseModel):

    messages: Annotated[List, add_messages]

    query: str

    search_result: Optional[str] = None

    retrieved_memories: Optional[list[dict]] = None

    context: Optional[str] = None

    final_result: Optional[str] = None


