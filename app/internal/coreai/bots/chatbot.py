from app.config import settings
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from app.internal.coreai.bots.chain import get_chain, ChainType
from langchain_core.messages import SystemMessage, trim_messages
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings

session_store = {}

embedding = HuggingFaceEmbeddings(model_name=settings.EMBEDDINGS_MODEL)

model = ChatGroq(model=settings.AI_MODEL, groq_api_key=settings.GROQ_API_KEY)


class Chatbot:
    INPUT_MESSAGES_KEY = "messages"

    @staticmethod
    def get_configuration():
        return {"configurable": {"session_id": "chat2"}}

    @staticmethod
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in session_store:
            session_store[session_id] = ChatMessageHistory()
        return session_store[session_id]

    @staticmethod
    def _get_with_messages_history(model):
        return RunnableWithMessageHistory(
            model,
            Chatbot.get_session_history,
            input_messages_key=Chatbot.INPUT_MESSAGES_KEY,
        )

    @staticmethod
    def translate(input: str, target_language: str = "vi"):
        return (
            get_chain(ChainType.TRANSLATE, model)
            .invoke(
                {
                    "target_language": target_language,
                    Chatbot.INPUT_MESSAGES_KEY: [("user", input)],
                }
            )
            .content
        )

    @staticmethod
    def chat(input: str):
        return (
            Chatbot._get_with_messages_history(get_chain(ChainType.CHAT, model))
            .invoke(
                {Chatbot.INPUT_MESSAGES_KEY: [HumanMessage(content=input)]},
                config=Chatbot.get_configuration(),
            )
            .content
        )
