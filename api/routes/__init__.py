from fastapi import APIRouter
from .embeddings import router as embeddings_router

# Criar um roteador principal
router = APIRouter()

# Incluir roteadores das rotas individuais
router.include_router(embeddings_router, prefix="/questions", tags=["questions"])
