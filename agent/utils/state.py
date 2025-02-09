from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage
from typing import TypedDict, Annotated, Sequence, List

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]  # Use LangChain message types
    current_word: dict
    current_step: str  # Track where we are in the lesson
    pronunciation_attempts: int
    meaning_attempts: int
    remaining_words: List[dict]
    last_user_input: str
    decision: str      # Router's decision
    output: str       # Response to show user
    disruption_history: List[dict]