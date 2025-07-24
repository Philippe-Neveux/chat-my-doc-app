import chainlit as cl

def get_conversation_history_with_current_prompt(content: str) -> str:
    """
    Retrieve the conversation history from the user session.
    If no history exists, return an empty list.
    """
    # Get conversation history from user session
    conversation_history = cl.user_session.get("conversation_history") or []
    
    conversation_history.append(f"Human: {content}")
    
    # Build conversation context
    conversation_context = "\n".join(conversation_history)
    
    # If this is the first message, don't include empty history
    return (
        content
        if len(conversation_history) == 1
        else f"{conversation_context}\nAssistant:"
    )

def update_conversation_history_from_llm_content(llm_content: str):
    conversation_history = cl.user_session.get("conversation_history") or []
    
    conversation_history.append(f"Assistant: {llm_content}")
    
    cl.user_session.set("conversation_history", conversation_history)