from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from app.config import settings
import os
import tempfile
import numpy as np
from typing import List, Dict, Any, Optional

VECTOR_STORE_DIR = os.path.join(tempfile.gettempdir(), "vectorstore")
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)


class RAGModel:
    def __init__(self, embed_model_name=settings.EMBEDDINGS_MODEL):
        self.embeddings = HuggingFaceEmbeddings(model_name=embed_model_name)
        self.vector_store = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        self.llm = ChatGroq(model=settings.AI_MODEL, groq_api_key=settings.GROQ_API_KEY)

    def initialize_from_messages(
        self, messages: List[Dict[str, Any]], conversation_id: str
    ):
        docs = []

        for message in messages:
            if (
                message["type"] == "text"
                and "content" in message
                and "text" in message["content"]
            ):
                text = message["content"]["text"]
                if not text or len(text.strip()) < 5:
                    continue

                sender = "AI" if message.get("isBot", False) else "User"
                if isinstance(message.get("sender"), dict) and message["sender"].get(
                    "userName"
                ):
                    sender = message["sender"].get("userName")

                doc = Document(
                    page_content=text,
                    metadata={
                        "sender": sender,
                        "timestamp": str(message.get("createdAt", "")),
                        "message_id": str(message.get("_id", "")),
                        "conversation_id": conversation_id,
                    },
                )
                docs.append(doc)

        if docs:
            chunked_docs = self.text_splitter.split_documents(docs)
            persist_directory = os.path.join(VECTOR_STORE_DIR, conversation_id)

            print("Persist directory: ", persist_directory)

            if os.path.exists(persist_directory):
                self.vector_store = Chroma(
                    persist_directory=persist_directory,
                    embedding_function=self.embeddings,
                )
                self.vector_store.add_documents(chunked_docs)
            else:
                self.vector_store = Chroma.from_documents(
                    documents=chunked_docs,
                    embedding=self.embeddings,
                    persist_directory=persist_directory,
                )

            self.vector_store.persist()
            return True

        return False

    def get_relevant_context(self, query: str, conversation_id: str, k: int = 5) -> str:
        persist_directory = os.path.join(VECTOR_STORE_DIR, conversation_id)

        if os.path.exists(persist_directory):
            if (
                not self.vector_store
                or self.vector_store._persist_directory != persist_directory
            ):
                self.vector_store = Chroma(
                    persist_directory=persist_directory,
                    embedding_function=self.embeddings,
                )

            docs = self.vector_store.similarity_search(query, k=k)

            context_lines = []
            for doc in docs:
                sender = doc.metadata.get("sender", "User")
                context_lines.append(f"{sender}: {doc.page_content}")

            return "\n".join(context_lines)

        return ""

    def get_rag_response(
        self, query: str, conversation_id: str, additional_context: str = ""
    ) -> str:
        retrieved_context = self.get_relevant_context(query, conversation_id)

        full_context = retrieved_context
        if additional_context:
            full_context = f"{additional_context}\n\n{retrieved_context}"

        if full_context:
            prompt = f"""You are a helpful AI assistant in a chat application.
            
Recent conversation context:
{full_context}

User's current question: {query}

Based on the conversation context above, please provide a helpful, relevant response. If the context doesn't contain relevant information, you can respond based on your general knowledge.
"""
        else:
            prompt = f"""You are a helpful AI assistant in a chat application.

User's question: {query}

Please provide a helpful, relevant response.
"""

        response = self.llm.invoke(prompt)
        return response.content


_rag_model = None


def get_rag_model():
    global _rag_model
    if _rag_model is None:
        _rag_model = RAGModel()
    return _rag_model
