import logging

import uvicorn
from fastapi import FastAPI

from app.api.routes import router
from app.config import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Text Summarization API",
    description="A fast, scalable text summarization service with support for Russian and English languages",
    version="1.0.0",
)

# Include API routes
app.include_router(router, prefix="/api/v1")


@app.get("/")
async def root():
    return {"message": "Text Summarization API", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
