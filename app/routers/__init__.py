from app.routers.chatbot import chatbot_router
from app.routers.summarize import summarize_router
from fastapi import APIRouter

app_router = APIRouter(prefix="/api/v1")

app_router.include_router(summarize_router)
app_router.include_router(chatbot_router)
