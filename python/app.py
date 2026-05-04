"""FastAPI entrypoint for the misinformation detection MVP backend."""

from fastapi import FastAPI

from api.routes import router as api_router

app = FastAPI(title="Misinfo RAG API", version="0.1.0")

# Register API routes.
app.include_router(api_router)
