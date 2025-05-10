from fastapi import APIRouter

summarize_router = APIRouter(prefix="/summarize", tags=["summarize"])


@summarize_router.get("/")
async def summarize():
    return {"message": "abc"}
