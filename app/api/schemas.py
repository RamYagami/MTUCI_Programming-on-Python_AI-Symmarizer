from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class SummarizationMode(str, Enum):
    extractive = "extractive"
    extractive_hierarchical = "extractive_hierarchical"
    abstractive = "abstractive"


class TextLength(str, Enum):
    short = "short"
    medium = "medium"
    long = "long"


class SummarizeRequest(BaseModel):
    text: str = Field(..., description="Input text to summarize")
    language: Optional[str] = Field(
        ...,
        pattern=r"^(ru|en)$",
        description="Language of the text (ru or en). If not provided, will be auto-detected.",
    )
    mode: SummarizationMode = Field(
        default=SummarizationMode.extractive, description="Summarization mode"
    )
    length: TextLength = Field(
        default=TextLength.medium, description="Desired length of summary"
    )


class SummarizeFileRequest(BaseModel):
    language: Optional[str] = Field(
        ...,
        pattern=r"^(ru|en)$",
        description="Language of the text (ru or en). If not provided, will be auto-detected.",
    )
    mode: SummarizationMode = Field(
        default=SummarizationMode.extractive, description="Summarization mode"
    )
    length: TextLength = Field(
        default=TextLength.medium, description="Desired length of summary"
    )


class ResultStatus(str, Enum):
    queued = "queued"
    processing = "processing"
    finished = "finished"
    failed = "failed"


class TaskResponse(BaseModel):
    task_id: str
    status: str = "queued"
    position_in_queue: int = Field(
        ..., description="Current position in the processing queue (1 = next)"
    )


class ResultResponse(BaseModel):
    status: ResultStatus
    summary: Optional[str] = None
    error: Optional[str] = None
    language: Optional[str] = None
    mode: Optional[SummarizationMode] = None
    length: Optional[TextLength] = None
    position_in_queue: Optional[int] = Field(
        None,
        description="Current position in queue (only shown when status is 'queued')",
    )
