from pydantic import BaseModel, Field
from typing import Optional, List


class Message(BaseModel):
    role: str = Field(..., description="User or assistant")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        title="User message",
        description="The message from the user to respond to",
    )
    conversation_id: Optional[str] = Field(
        None,
        title="Conversation ID",
        description="ID of the conversation to retrieve context from",
    )
    user_id: Optional[str] = Field(
        None,
        title="User ID",
        description="ID of the user sending the message",
    )
    refresh_context: Optional[bool] = Field(
        False,
        title="Refresh Context",
        description="Whether to force refresh the context",
    )
    context: Optional[str] = Field(
        None,
        title="Conversation Context",
        description="The context of the conversation",
    )


class ChatResponse(BaseModel):
    reply: str = Field(
        ...,
        title="Chat response",
        description="The chatbot's response to the user's input text",
    )
    conversation_id: Optional[str] = Field(
        None,
        title="Conversation ID",
        description="ID of the conversation this response belongs to",
    )
    sources: Optional[List[str]] = Field(
        None,
        title="Sources",
        description="Sources of information used for the response",
    )


class TranslationRequest(BaseModel):
    text: str = Field(
        ...,
        title="Text to translate",
        description="The text to be translated",
    )
    target_language: str = Field(
        "vi",
        title="Target language",
        description="The language to translate to (default: vi)",
    )


class TranslationResponse(BaseModel):
    translated: str = Field(
        ...,
        title="Translated response",
        description="The translated result of the user's input text",
    )
