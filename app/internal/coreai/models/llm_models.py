from app.config import settings
from langchain_community.llms import Ollama


def get_llm(model_name=settings.AI_MODEL):
    return Ollama(model_name=model_name)
