from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from app.schemas.audio import AudioSummarizeRequest, AudioSummarizeResponse
from app.internal.audio_utils import (
    download_from_s3,
    transcribe_audio,
    generate_summary,
    clean_up_temp_file,
    transcribe_audio_groq,
)
import os
from typing import Dict, Any

summarize_router = APIRouter(prefix="/summarize", tags=["summarize"])


@summarize_router.get("/")
async def test_summarize():
    """Test endpoint to check if the summarization API is working."""
    return {"message": "Summarization API is working"}


@summarize_router.get("/transcription")
async def get_transcription(file_path: str):
    temp_file_path = download_from_s3(file_path)
    transcription_result = transcribe_audio_groq(
        audio_file_path=temp_file_path, language="vi"
    )
    return transcription_result


@summarize_router.post("/audio", response_model=AudioSummarizeResponse)
async def summarize_audio(
    request: AudioSummarizeRequest, background_tasks: BackgroundTasks
):
    try:
        temp_file_path = download_from_s3(request.file_path)

        background_tasks.add_task(clean_up_temp_file, temp_file_path)

        import librosa

        duration = librosa.get_duration(path=temp_file_path)

        transcription_result = transcribe_audio_groq(
            audio_file_path=temp_file_path, language=request.language
        )

        summary = generate_summary(
            text=transcription_result,
            max_length=request.max_length,
            language=request.language,
        )

        return AudioSummarizeResponse(
            summary=summary,
            transcription=transcription_result,
            duration=duration,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")
