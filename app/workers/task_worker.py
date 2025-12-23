import logging

from app.config import config
from app.models.summarizer import TextSummarizer
from redis import Redis

logger = logging.getLogger(__name__)
summarizer = TextSummarizer()


def summarize_text_task(
    task_id: str,
    text: str,
    language: str,
    mode: str,
    length: str,
):
    """Background task to perform text summarization"""
    try:
        redis_conn = Redis(
            host=config.REDIS_HOST, port=config.REDIS_PORT, db=config.REDIS_DB
        )

        redis_conn.hset(f"task:{task_id}", "status", "processing")

        if mode == "extractive":
            summary = summarizer.extractive_summarize(
                text,
                language,
                length,
            )

        elif mode == "extractive_hierarchical":
            summary = summarizer.extractive_hierarchical_summarize(
                text,
                language,
                length,
            )

        elif mode == "abstractive":
            summary = summarizer.abstractive_summarize(
                text,
                language,
                length,
            )

        else:
            raise ValueError(f"Unknown mode: {mode}")

        redis_conn.setex(f"result:{task_id}", config.RESULT_TTL, summary)
        redis_conn.hset(f"task:{task_id}", "status", "finished")

        logger.info(f"Task {task_id} completed successfully")

    except Exception as e:
        logger.error(f"Task {task_id} failed: {e}")

        redis_conn = Redis(
            host=config.REDIS_HOST, port=config.REDIS_PORT, db=config.REDIS_DB
        )
        redis_conn.hset(f"task:{task_id}", "status", "failed")
        redis_conn.hset(f"task:{task_id}", "error", str(e))
