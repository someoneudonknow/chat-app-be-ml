from app.internal.coreai.prompts.chat_prompts import chat_prompt, translator_prompt
from enum import Enum
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import trim_messages


class ChainType(Enum):
    CHAT = "chat"
    TRANSLATE = "translate"


chain_map = {
    ChainType.CHAT: lambda model: RunnablePassthrough.assign(
        messages=itemgetter("messages")
        | trim_messages(
            max_tokens=8000,
            strategy="last",
            token_counter=model,
            allow_partial=False,
            start_on="human",
            include_system=True,
        )
    )
    | chat_prompt
    | model,
    ChainType.TRANSLATE: lambda model: translator_prompt | model,
}


def get_chain(chain_type: ChainType, model):
    if chain_type not in chain_map:
        raise ValueError(f"Chain type {type} not found.")

    return chain_map[chain_type](model)
