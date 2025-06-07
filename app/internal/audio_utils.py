import os
import tempfile
import boto3
from botocore.exceptions import ClientError
import whisper
from app.config import settings
import time
from groq import Groq
from pydub import AudioSegment
import noisereduce as nr
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import lru_cache
import logging
from typing import Optional, Dict, Any
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_whisper_model_cache = {}
_groq_client_cache = None


@lru_cache(maxsize=1)
def get_whisper_model(model_name: str = None):
    """Cache Whisper model to avoid reloading"""
    model_name = model_name or settings.WHISPER_MODEL
    if model_name not in _whisper_model_cache:
        logger.info(f"Loading Whisper model: {model_name}")
        _whisper_model_cache[model_name] = whisper.load_model(model_name)
    return _whisper_model_cache[model_name]


def get_groq_client():
    """Cache Groq client"""
    global _groq_client_cache
    if _groq_client_cache is None:
        _groq_client_cache = Groq(api_key=settings.GROQ_API_KEY)
    return _groq_client_cache


async def download_from_s3_async(file_path: str) -> str:
    """Async S3 download using aiobotocore or threading"""
    try:

        def _download():
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

        # Run S3 download in thread pool
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            temp_file_path = await loop.run_in_executor(executor, _download)

        return temp_file_path

    except ClientError as e:
        raise Exception(f"Error downloading file from S3: {str(e)}")


def download_from_s3(file_path: str) -> str:
    """Synchronous version with connection pooling"""
    try:
        # Reuse S3 client with connection pooling
        if not hasattr(download_from_s3, "_s3_client"):
            download_from_s3._s3_client = boto3.client(
                "s3",
                aws_access_key_id=settings.AWS_ACCESS_KEY,
                aws_secret_access_key=settings.AWS_SECRET_KEY,
                region_name=settings.AWS_REGION,
                config=boto3.session.Config(
                    max_pool_connections=50, retries={"max_attempts": 3}
                ),
            )

        temp_file = tempfile.NamedTemporaryFile(
            delete=False, suffix=os.path.splitext(file_path)[1]
        )
        temp_file_path = temp_file.name
        temp_file.close()

        download_from_s3._s3_client.download_file(
            settings.AWS_BUCKET_NAME, file_path, temp_file_path
        )

        return temp_file_path

    except ClientError as e:
        raise Exception(f"Error downloading file from S3: {str(e)}")


def get_audio_file_hash(file_path: str) -> str:
    """Generate hash for audio file caching"""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        # Read first and last 8KB for quick hash
        hasher.update(f.read(8192))
        f.seek(-8192, 2)
        hasher.update(f.read(8192))
    return hasher.hexdigest()


def smart_preprocess_audio(
    input_path: str,
    output_path: str,
    sample_rate: int = 16000,
    target_duration: Optional[float] = None,
    quick_mode: bool = False,
) -> str:
    """Optimized audio preprocessing with smart defaults"""
    try:
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        start_time = time.time()
        audio = AudioSegment.from_file(input_path)

        # Quick duration check - skip processing very short files
        duration_ms = len(audio)
        if duration_ms < 1000:  # Less than 1 second
            logger.info("Audio too short, skipping preprocessing")
            audio.export(output_path, format="wav")
            return output_path

        # Limit processing for very long files
        if target_duration and duration_ms > target_duration * 1000:
            logger.info(f"Truncating audio to {target_duration} seconds")
            audio = audio[: int(target_duration * 1000)]

        # Basic optimization - convert to mono and resample
        if audio.channels > 1:
            audio = audio.set_channels(1)

        if audio.frame_rate != sample_rate:
            audio = audio.set_frame_rate(sample_rate)

        # Skip heavy processing in quick mode
        if not quick_mode:
            # Normalize audio
            if audio.max_dBFS > -0.1:
                logger.warning("Clipping detected, normalizing audio")
            audio = audio.apply_gain(-audio.max_dBFS)

            # Noise reduction (expensive operation)
            audio_samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            if len(audio_samples) > 0:
                reduced_noise = nr.reduce_noise(
                    y=audio_samples, sr=sample_rate, stationary=True
                )
                audio = AudioSegment(
                    reduced_noise.astype(np.int16).tobytes(),
                    frame_rate=sample_rate,
                    sample_width=2,
                    channels=1,
                )

            # Apply filters
            audio = audio.low_pass_filter(8000).high_pass_filter(80)

            # Remove silence
            silence_thresh = max(-40, audio.dBFS - 10)
            audio = audio.strip_silence(silence_len=100, silence_thresh=silence_thresh)

        # Export optimized audio
        audio.export(
            output_path, format="wav", parameters=["-ac", "1", "-ar", str(sample_rate)]
        )

        processing_time = time.time() - start_time
        logger.info(f"Audio preprocessing completed in {processing_time:.2f}s")

        return output_path

    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        raise ValueError(f"Failed to process audio file: {str(e)}")


async def transcribe_audio_groq_async(
    audio_file_path: str,
    language: str = "vi",
    preprocess: bool = True,
    quick_mode: bool = False,
) -> str:
    """Async Groq transcription with optimizations"""
    try:
        if preprocess:
            preprocessed_path = audio_file_path.replace(
                os.path.splitext(audio_file_path)[1], "_preprocessed.wav"
            )

            # Run preprocessing in thread pool
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                await loop.run_in_executor(
                    executor,
                    smart_preprocess_audio,
                    audio_file_path,
                    preprocessed_path,
                    16000,
                    None,
                    quick_mode,
                )
            audio_path = preprocessed_path
        else:
            audio_path = audio_file_path

        # Async file reading and API call
        groq = get_groq_client()

        def _transcribe():
            with open(audio_path, "rb") as audio_file:
                response = groq.audio.transcriptions.create(
                    model="whisper-large-v3-turbo",
                    file=audio_file,
                    language=language,
                )
                return response.text

        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            result = await loop.run_in_executor(executor, _transcribe)

        # Cleanup preprocessed file
        if preprocess and os.path.exists(audio_path):
            os.remove(audio_path)

        return result

    except Exception as e:
        raise Exception(f"Error transcribing audio: {str(e)}")


def transcribe_audio_groq_optimized(
    audio_file_path: str,
    language: str = "vi",
    preprocess: bool = True,
    quick_mode: bool = False,
) -> str:
    """Optimized Groq transcription"""
    try:
        if preprocess:
            preprocessed_path = audio_file_path.replace(
                os.path.splitext(audio_file_path)[1], "_preprocessed.wav"
            )
            smart_preprocess_audio(
                audio_file_path, preprocessed_path, quick_mode=quick_mode
            )
            audio_path = preprocessed_path
        else:
            audio_path = audio_file_path

        groq = get_groq_client()

        with open(audio_path, "rb") as audio_file:
            response = groq.audio.transcriptions.create(
                model="whisper-large-v3-turbo",
                file=audio_file,
                language=language,
            )

        if preprocess and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except:
                pass

        return response.text

    except Exception as e:
        raise Exception(f"Error transcribing audio: {str(e)}")


def transcribe_audio_optimized(
    audio_file_path: str, language: str = "vi", model_size: str = None
) -> Dict[str, Any]:
    """Optimized local Whisper transcription"""
    try:
        # Use smaller model for faster processing if not specified
        model_size = model_size or getattr(settings, "WHISPER_MODEL", "base")
        model = get_whisper_model(model_size)

        start_time = time.time()

        # Optimized transcription parameters
        result = model.transcribe(
            audio_file_path,
            language=language,
            fp16=False,
            beam_size=1,  # Faster beam search
            best_of=1,  # Single pass
            temperature=0.0,  # Deterministic
            condition_on_previous_text=False,  # Faster processing
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


async def generate_summary_async(
    text: str, max_length: int = 500, language: str = "en"
) -> str:
    """Async summary generation"""
    try:
        from langchain.prompts import PromptTemplate
        from langchain_groq import ChatGroq

        language_instructions = {
            "en": "English.",
            "vi": "tiếng Việt.",
        }

        language_instruction = language_instructions.get(
            language, f"Summarize in {language}."
        )

        summary_template = """Summarize the key points from this transcription in {language_instruction}

        Text: {text}
        
        Limit: {max_length} characters. Focus on main topics and conclusions only."""

        llm = ChatGroq(
            api_key=settings.GROQ_API_KEY,
            model_name="llama3-8b-8192",
            temperature=0.1,
        )

        prompt = PromptTemplate(
            input_variables=["text", "max_length", "language_instruction"],
            template=summary_template,
        )

        def _generate():
            chain = prompt | llm
            result = chain.invoke(
                {
                    "text": text[:4000],
                    "max_length": max_length,
                    "language_instruction": language_instruction,
                }
            )
            return result.content.strip()

        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            summary = await loop.run_in_executor(executor, _generate)

        return summary

    except Exception as e:
        raise Exception(f"Error generating summary: {str(e)}")


async def process_audio_pipeline_async(
    file_path: str,
    language: str = "vi",
    max_summary_length: int = 500,
    use_groq: bool = True,
    quick_mode: bool = False,
) -> Dict[str, Any]:
    start_time = time.time()
    temp_files = []

    try:
        if file_path.startswith("s3://") or not os.path.exists(file_path):
            logger.info("Downloading from S3...")
            local_file_path = await download_from_s3_async(file_path)
            temp_files.append(local_file_path)
        else:
            local_file_path = file_path

        logger.info("Starting transcription...")
        if use_groq:
            transcript = await transcribe_audio_groq_async(
                local_file_path, language, quick_mode=quick_mode
            )
        else:
            loop = asyncio.get_event_loop()
            with ProcessPoolExecutor() as executor:
                result = await loop.run_in_executor(
                    executor, transcribe_audio_optimized, local_file_path, language
                )
            transcript = result["text"]

        logger.info("Generating summary...")
        summary_task = asyncio.create_task(
            generate_summary_async(transcript, max_summary_length, language)
        )

        summary = await summary_task

        total_time = time.time() - start_time

        return {
            "transcript": transcript,
            "summary": summary,
            "language": language,
            "processing_time": total_time,
            "method": "groq" if use_groq else "local",
        }

    except Exception as e:
        raise Exception(f"Pipeline error: {str(e)}")
    finally:
        for temp_file in temp_files:
            clean_up_temp_file(temp_file)


def clean_up_temp_file(file_path: str) -> None:
    """Enhanced cleanup with better error handling"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.debug(f"Cleaned up temp file: {file_path}")
    except Exception as e:
        logger.warning(f"Failed to delete temporary file {file_path}: {str(e)}")


def process_audio_fast(
    file_path: str,
    language: str = "vi",
    max_summary_length: int = 500,
    quick_mode: bool = True,
) -> Dict[str, Any]:
    """Fast processing with reasonable defaults"""
    return asyncio.run(
        process_audio_pipeline_async(
            file_path,
            language,
            max_summary_length,
            use_groq=True,
            quick_mode=quick_mode,
        )
    )
