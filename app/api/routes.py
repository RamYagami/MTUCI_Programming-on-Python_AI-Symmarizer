import logging
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from redis import Redis
from rq import Queue
from tika_client import AsyncTikaClient

from app.api.schemas import (
    ResultResponse,
    ResultStatus,
    SummarizationMode,
    SummarizeFileRequest,
    SummarizeRequest,
    TaskResponse,
    TextLength,
)
from app.config import config
from app.models.summarizer import TextSummarizer
from app.workers.task_worker import summarize_text_task

logger = logging.getLogger(__name__)

router = APIRouter()

tika_client = AsyncTikaClient(
    tika_url=config.TIKA_URL,
    timeout=config.TIKA_TIMEOUT,
)

summarizer = TextSummarizer()

redis_conn = Redis(
    host=config.REDIS_HOST,
    port=config.REDIS_PORT,
    db=config.REDIS_DB,
)
q = Queue(connection=redis_conn)


@router.post("/summarize", response_model=TaskResponse)
async def submit_summarization(request: SummarizeRequest):
    try:
        return _enqueue_summarization_task(
            request.text,
            request.language,
            request.mode,
            request.length,
        )

    except Exception as e:
        logger.error(f"Error submitting summarization task: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


def _get_summarize_file_params(
    language: Optional[str] = Form(None),
    mode: str = Form("normal"),
    length: str = Form("medium"),
) -> SummarizeFileRequest:
    return SummarizeFileRequest(
        language=language,
        mode=SummarizationMode(mode),
        length=TextLength(length),
    )


@router.post("/summarize/file", response_model=TaskResponse)
async def submit_summarization_file(
    file: UploadFile = File(...),
    params: SummarizeFileRequest = Depends(_get_summarize_file_params),
):
    if file.size is None or file.size > config.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"""File too large. Maximum allowed size: {
                config.MAX_FILE_SIZE // (1024 * 1024)
            } MB.""",
        )

    file_bytes = await file.read()
    content_type = file.content_type or "application/octet-stream"
    logger.info(
        f"""Processing file '{file.filename}' ({content_type}), size: {
            file.size
        } bytes"""
    )

    text = await _extract_text_from_file(file_bytes, content_type)

    try:
        return _enqueue_summarization_task(
            text=text,
            language=params.language,
            mode=params.mode,
            length=params.length,
        )

    except Exception as e:
        logger.error(f"Failed to enqueue summarization task: {e}")
        raise HTTPException(
            status_code=500, detail="Internal error while submitting task."
        )


@router.get("/result/{task_id}", response_model=ResultResponse)
async def get_result(task_id: str):
    try:
        task_key = f"task:{task_id}"
        task_data = redis_conn.hgetall(task_key)

        if not task_data:
            raise HTTPException(status_code=404, detail="Task not found")

        task_data = {k.decode(): v.decode() for k, v in task_data.items()}
        status = task_data.get("status", "unknown")

        position_in_queue = None
        if status == "queued":
            position_in_queue = _get_position_in_queue(task_id, q)

        if status == "finished":
            summary = redis_conn.get(f"result:{task_id}")

            summary = summary.decode() if summary else None

            return ResultResponse(
                status=ResultStatus.finished,
                summary=summary,
                language=task_data.get("language"),
                mode=task_data.get("mode"),
                length=task_data.get("length"),
                position_in_queue=position_in_queue,
            )

        elif status == "failed":
            return ResultResponse(
                status=ResultStatus.failed,
                error=task_data.get("error", "Unknown error"),
                position_in_queue=position_in_queue,
            )

        elif status == "processing":
            return ResultResponse(
                status=ResultStatus.processing, position_in_queue=None
            )

        else:
            return ResultResponse(
                status=ResultStatus.queued,
                language=task_data.get("language"),
                mode=task_data.get("mode"),
                length=task_data.get("length"),
                position_in_queue=position_in_queue,
            )

    except Exception as e:
        logger.error(f"Error getting result for task {task_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


async def _extract_text_from_file(
    file_bytes: bytes,
    content_type: str,
) -> str:
    try:
        response = await tika_client.tika.as_text.from_buffer(
            file_bytes,
            content_type,
        )
        text = response.content

        if not text:
            logger.warning("Extracted text is empty")
            return ""

        return text

    except Exception as e:
        logger.error(f"Tika extraction failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Text extraction from file is failed",
        )


def _enqueue_summarization_task(
    text: str,
    language: Optional[str],
    mode: SummarizationMode,
    length: TextLength,
) -> TaskResponse:
    if len(text) > config.MAX_TEXT_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"""Text too long. Maximum allowed: {
                config.MAX_TEXT_LENGTH
            } characters.""",
        )

    if not language:
        language = summarizer.detect_language(text)

    task_id = str(uuid4())
    task_info = {
        "text": text,
        "language": language,
        "mode": mode,
        "length": length,
        "status": "queued",
    }

    redis_conn.hmset(f"task:{task_id}", task_info)
    redis_conn.expire(f"task:{task_id}", config.MAX_TASK_LIFETIME)

    q.enqueue(
        summarize_text_task,
        task_id=task_id,
        text=text,
        language=language,
        mode=mode.value,
        length=length.value,
        job_timeout=config.MAX_TASK_LIFETIME,
        job_id=task_id,
    )

    position = _get_position_in_queue(task_id, q) or 1
    logger.info(f"Task {task_id} submitted. Position in queue: {position}")

    return TaskResponse(
        task_id=task_id,
        status="queued",
        position_in_queue=position,
    )


def _get_position_in_queue(task_id: str, queue: Queue) -> Optional[int]:
    """
    Returns 1-based position of task_id in the queue.
    Returns None if task is not in queue (already started or finished).
    """
    try:
        job_ids = queue.job_ids  # list of job IDs in order
        if task_id in job_ids:
            return job_ids.index(task_id) + 1
        return None
    except Exception as e:
        logger.warning(f"Failed to get position in queue for {task_id}: {e}")
        return None
