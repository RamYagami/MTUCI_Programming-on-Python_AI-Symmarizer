import os


class Config:
    REDIS_HOST: str = os.getenv("REDIS_HOST", "redis")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", 6379))
    REDIS_DB: int = int(os.getenv("REDIS_DB", 0))
    RESULT_TTL: int = int(os.getenv("RESULT_TTL", 3600))  # 1 hour
    MAX_TEXT_LENGTH: int = int(os.getenv("MAX_TEXT_LENGTH", 100_000))
    MAX_TASK_LIFETIME: int = int(
        os.getenv("MAX_TASK_LIFETIME", 300))  # 5 minutes
    TIKA_URL: str = os.getenv("TIKA_URL", "http://tika:9998")
    TIKA_TIMEOUT = int(os.getenv("TIKA_TIMEOUT", 60))  # 60 seconds
    MAX_FILE_SIZE = int(os.getenv("TIKA_TIMEOUT", 10 * 1024 * 1024))  # 10 MB


config = Config()
