from fastapi import FastAPI
from routes import router as api_router
from services.embedding_service import init_db


init_db()
app = FastAPI()

# Incluir o roteador principal
app.include_router(api_router)

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)