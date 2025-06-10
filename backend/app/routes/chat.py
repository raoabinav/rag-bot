"""
Chat API routes.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ..services.chat_service import chat_service

router = APIRouter()

class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str

@router.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Handle chat messages.
    Args:
        request: The chat request containing the user's message
    Returns:
        Response containing the assistant's reply
    """
    try:
        # Process the message using the chat service
        response = chat_service.process_chat_message(
            message=request.message,
            namespace="avengers-bot"  
        )
        return {"response": response}
    except Exception as e:
        # Log the error
        print(f"Error in chat endpoint: {e}")
        
        # Return a user-friendly error message
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your message. Please try again."
        )
