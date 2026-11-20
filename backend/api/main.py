"""
Nash backend API.

Run with:
    uvicorn backend.api.main:app --host 0.0.0.0 --port 8000

Every route lives under /api, because nginx forwards /api/... to this server unchanged
(see deployment/nginx/nginx.conf) and docker-compose checks /api/health.
"""
from fastapi import APIRouter, FastAPI

from backend.configs.tools.retrieval import router as tools_router
from backend.pipelines.rag.rag_pipeline import router as rag_router

app = FastAPI(title="Nash API")

api = APIRouter(prefix="/api")
api.include_router(rag_router)      # /api/rag/...
api.include_router(tools_router)    # /api/tools/...


@api.get("/health")
def health():
    return {"status": "ok"}


app.include_router(api)
