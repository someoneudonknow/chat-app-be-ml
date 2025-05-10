from app.internal.coreai.bots.rag_chatbot import get_rag_chatbot
from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    TranslationRequest,
    TranslationResponse,
)
from fastapi import APIRouter, Query, Body, HTTPException, Depends

chatbot_router = APIRouter(prefix="/chatbot", tags=["chatbot"])


async def get_chatbot():
    return get_rag_chatbot()


@chatbot_router.post("/chat", response_model=ChatResponse)
async def chat_handler(request: ChatRequest = Body(...), chatbot=Depends(get_chatbot)):
    try:
        if request.context:
            reply = await chatbot.chat(
                user_message=request.message,
                conversation_id=request.conversation_id,
                user_id=request.user_id,
                context=request.context,
            )
        else:
            reply = await chatbot.chat(
                user_message=request.message,
                conversation_id=request.conversation_id,
                user_id=request.user_id,
            )

        return {"reply": reply, "conversation_id": request.conversation_id}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing chat request: {str(e)}"
        )


@chatbot_router.post("/translate", response_model=TranslationResponse)
async def translate_handler(
    request: TranslationRequest = Body(...), chatbot=Depends(get_chatbot)
):
    try:
        translated = await chatbot.translate(
            text=request.text, target_language=request.target_language
        )

        return {"translated": translated}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error translating text: {str(e)}")
