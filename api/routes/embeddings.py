from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.embedding_service import get_similar_questions_from_db
from services.question_service import insert_question


# Inicializando o FastAPI
router = APIRouter()



# Modelo de entrada para a API
class QuestionInput(BaseModel):
    question: str

# Endpoint para buscar as top-N questões semelhantes e refiná-las com o Cross-Encoder
@router.post("/similar-questions/")
async def get_similar_questions(input: QuestionInput):

    pairs_with_scores = get_similar_questions_from_db(input.question)


# Convertendo numpy.float32 para float nativo
    result = [{"question": pair, "similarity_score": float(score)} for pair, score in pairs_with_scores if score > 0]
    return {"similar_questions": result}




@router.post("/add-new-question/")
async def add_new_question(input: QuestionInput):
    
    try:
        # Supondo que `input` tenha um atributo `question`
        insert_question(input.question)
        return {"status": "success", "message": "Questão adicionada com sucesso."}
    except Exception as e:
        print(f"Erro ao adicionar questão: {e}")
        raise HTTPException(status_code=500, detail=f"Erro ao adicionar questão: {str(e)}")