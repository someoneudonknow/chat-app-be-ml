from pydantic import BaseModel, Field


class AudioSummarizeRequest(BaseModel):
    file_path: str = Field(..., description="S3 file path of the audio to summarize")
    max_length: int = Field(
        default=500, description="Maximum length of the summary in characters"
    )
    language: str = Field(default="en", description="Language of the audio content")


class AudioSummarizeResponse(BaseModel):
    summary: str = Field(..., description="Generated summary of the audio content")
    transcription: str = Field(
        ..., description="Full transcription of the audio content"
    )
    duration: float = Field(..., description="Duration of the audio in seconds")
