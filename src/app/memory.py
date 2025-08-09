from typing import List, Dict

from google.genai import types


def get_conversation_history(
    session_id: str,
    conversation_histories: Dict[str, List[types.Content]]
) -> List[types.Content]:
    """Get conversation history for a session."""
    if session_id not in conversation_histories:
        conversation_histories[session_id] = []
    return conversation_histories[session_id]

def add_to_history(
    session_id: str,
    role: str, 
    content: str,
    conversation_histories: Dict[str, List[types.Content]]
) -> None:
    """Add a message to conversation history."""
    history = get_conversation_history(session_id, conversation_histories)
    
    history.append(types.Content(
        role=role,
        parts=[types.Part(text=content)]
    ))
