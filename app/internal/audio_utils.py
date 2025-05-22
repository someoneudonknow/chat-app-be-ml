import os
import tempfile
import boto3
from botocore.exceptions import ClientError
import whisper
from app.config import settings
import time
from groq import Groq


def download_from_s3(file_path: str) -> str:
    try:
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=settings.AWS_ACCESS_KEY,
            aws_secret_access_key=settings.AWS_SECRET_KEY,
            region_name=settings.AWS_REGION,
        )

        temp_file = tempfile.NamedTemporaryFile(
            delete=False, suffix=os.path.splitext(file_path)[1]
        )
        temp_file_path = temp_file.name
        temp_file.close()

        s3_client.download_file(settings.AWS_BUCKET_NAME, file_path, temp_file_path)

        return temp_file_path

    except ClientError as e:
        raise Exception(f"Error downloading file from S3: {str(e)}")


def transcribe_audio_groq(audio_file_path: str, language: str = "vi") -> dict:
    try:
        groq = Groq(api_key=settings.GROQ_API_KEY)
        response = groq.audio.transcriptions.create(
            model="whisper-large-v3-turbo",
            file=open(audio_file_path, "rb"),
            language=language,
        )

        return response.text

    except Exception as e:
        raise Exception(f"Error transcribing audio: {str(e)}")


def transcribe_audio(audio_file_path: str, language: str = "vi") -> dict:
    try:
        model = whisper.load_model(settings.WHISPER_MODEL)

        start_time = time.time()
        result = model.transcribe(
            audio_file_path,
            language=language,
            fp16=False,
        )
        transcription_time = time.time() - start_time

        return {
            "text": result["text"],
            "segments": result["segments"],
            "language": result["language"],
            "processing_time": transcription_time,
        }

    except Exception as e:
        raise Exception(f"Error transcribing audio: {str(e)}")


def generate_summary(text: str, max_length: int = 500, language: str = "en") -> str:
    try:
        from langchain.prompts import PromptTemplate
        from langchain_groq import ChatGroq

        llm = ChatGroq(api_key=settings.GROQ_API_KEY, model_name="llama3-8b-8192")

        language_instructions = {
            "en": "Write the summary in English.",
            "vi": "Viết tóm tắt bằng tiếng Việt.",
        }

        language_instruction = language_instructions.get(
            language, f"Write the summary in {language} language."
        )

        summary_template = """
You are an expert at summarizing audio transcriptions.

Below is a transcription of an audio recording in a specific language.

{text}

Your task is to write a concise summary that captures the key points, main topics, and important conclusions from the transcription.

The summary must be written **only in the original language of the transcription**, which is: {language_instruction}

Do not translate to or include any other language.
The result should be only the pure summary in the target language.

Limit the length to a maximum of {max_length} characters.
"""

        prompt = PromptTemplate(
            input_variables=["text", "max_length", "language_instruction"],
            template=summary_template,
        )

        chain = prompt | llm
        result = chain.invoke(
            {
                "text": text,
                "max_length": max_length,
                "language_instruction": language_instruction,
            }
        )

        return result.content.strip()

    except Exception as e:
        raise Exception(f"Error generating summary: {str(e)}")


def clean_up_temp_file(file_path: str) -> None:
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        print(f"Warning: Failed to delete temporary file {file_path}: {str(e)}")
