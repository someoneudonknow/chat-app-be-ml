from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from app.schemas.audio import AudioSummarizeRequest, AudioSummarizeResponse
from app.internal.audio_utils import (
    download_from_s3_async,
    download_from_s3,
    generate_summary_async,
    clean_up_temp_file,
    transcribe_audio_groq_optimized,
)
import librosa
import os

summarize_router = APIRouter(prefix="/summarize", tags=["summarize"])


@summarize_router.get("/")
async def test_summarize():
    """Test endpoint to check if the summarization API is working."""
    return {"message": "Summarization API is working"}


@summarize_router.get("/transcription")
async def get_transcription(file_path: str):
    if file_path.startswith("s3://") or not os.path.exists(file_path):
        temp_file_path = await download_from_s3_async(file_path)
    else:
        temp_file_path = download_from_s3(file_path)

    transcription_result = transcribe_audio_groq_optimized(
        audio_file_path=temp_file_path, language="vi", preprocess=True, quick_mode=True
    )

    if temp_file_path != file_path:
        clean_up_temp_file(temp_file_path)

    return transcription_result


@summarize_router.post("/audio", response_model=AudioSummarizeResponse)
async def summarize_audio(
    request: AudioSummarizeRequest, background_tasks: BackgroundTasks
):
    try:
        if request.file_path.startswith("s3://") or not os.path.exists(
            request.file_path
        ):
            temp_file_path = await download_from_s3_async(request.file_path)
        else:
            temp_file_path = download_from_s3(request.file_path)

        background_tasks.add_task(clean_up_temp_file, temp_file_path)

        duration = librosa.get_duration(path=temp_file_path)

        transcription_result = transcribe_audio_groq_optimized(
            audio_file_path=temp_file_path,
            language=request.language,
            preprocess=True,
            quick_mode=True,
        )

        summary = await generate_summary_async(
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
