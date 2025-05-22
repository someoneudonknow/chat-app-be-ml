from langchain_groq import ChatGroq
from app.config import settings
from app.internal.coreai.models.rag_models import get_rag_model
from app.internal.coreai.data.message_repository import get_message_repository
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGChatbot:
    def __init__(self):
        self.rag_model = get_rag_model()
        self.message_repository = get_message_repository()
        self.llm = ChatGroq(model=settings.AI_MODEL, groq_api_key=settings.GROQ_API_KEY)

        self.chat_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant in a chat application. Provide concise, accurate, and helpful responses.",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

    async def initialize_context(
        self, conversation_id: str, force_refresh: bool = False
    ) -> bool:
        try:
            messages = self.message_repository.get_conservation_messages(
                conversation_id, limit=30
            )

            if not messages:
                logger.warning(f"No messages found for conversation {conversation_id}")
                return False

            result = self.rag_model.initialize_from_messages(messages, conversation_id)

            if result:
                logger.info(
                    f"Successfully initialized RAG context for conversation {conversation_id}"
                )
            else:
                logger.warning(
                    f"Failed to initialize RAG context for conversation {conversation_id}"
                )

            return result

        except Exception as e:
            logger.error(f"Error initializing context: {str(e)}")
            return False

    async def chat(
        self,
        user_message: str,
        conversation_id: str = None,
        user_id: str = None,
        context: str = None,
    ) -> str:
        try:
            if not conversation_id:
                response = self.llm.invoke(
                    self.chat_prompt.format(
                        messages=[HumanMessage(content=user_message)]
                    )
                )
                return response.content

            if context:
                logger.info(
                    f"Using provided context for conversation {conversation_id}"
                )
                prompt = f"""You are a helpful AI assistant in a chat application.
                
                Recent conversation history:
                {context}

                Please respond to the user's latest message based on this conversation history. Be helpful, concise, and informative.
                """
                response = self.llm.invoke(prompt)
                return response.content

            await self.initialize_context(conversation_id)

            conservation_info = self.message_repository.get_conservation_info(
                conversation_id
            )
            conservation_members = self.message_repository.get_conservation_members(
                conversation_id
            )

            additional_context = ""
            if conservation_info:
                conversation_type = conservation_info.get("type", "GROUP")
                member_count = conservation_info.get("member_count", 0)

                additional_context = f"This is a {conversation_type.lower()} conversation with {member_count} members."

                if conservation_members:
                    member_names = [
                        member.get("userName") or member.get("nickname")
                        for member in conservation_members
                    ]
                    additional_context += (
                        f" Participants include: {', '.join(member_names)}."
                    )

            response = self.rag_model.get_rag_response(
                query=user_message,
                conversation_id=conversation_id,
                additional_context=additional_context,
            )

            return response

        except Exception as e:
            logger.error(f"Error getting chat response: {str(e)}")
            return "I'm sorry, I'm having trouble processing your request right now. Please try again later."

    async def translate(self, text: str, target_language: str = "vi") -> str:
        try:
            translate_prompt = f"""Translate the following text to {target_language}. 
Provide only the translation without any additional explanations.

Text to translate: {text}"""

            response = self.llm.invoke(translate_prompt)
            return response.content

        except Exception as e:
            logger.error(f"Error translating text: {str(e)}")
            return "Translation error. Please try again later."


_rag_chatbot = None


def get_rag_chatbot():
    global _rag_chatbot
    if _rag_chatbot is None:
        _rag_chatbot = RAGChatbot()
    return _rag_chatbot
