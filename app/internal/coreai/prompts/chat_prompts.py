from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

translator_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "I want you to act as a language translator. I will speak to you in any language and you will detect the source language, translate it into {target_language}. Please enhance the translation by using more sophisticated vocabulary and elegant expressions while maintaining the original meaning. Provide only the translation without any additional explanations.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

chat_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer the following question to the nest of your ability.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

ask_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Please answer the following question.",
        ),
        ("user", "Question: {input}"),
    ]
)

summarize_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Please summarize the following text.",
        ),
        ("user", "Text: {input}"),
    ]
)
