import numpy as np
import psycopg2
from sentence_transformers import SentenceTransformer
from services.config import get_db_connection





# Chame a função para criar a tabela
#create_table()




def insert_question(question):
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # Carregar o modelo SBERT
        bi_encoder = SentenceTransformer("api/models/SBERT")

        # Codificar a questão em embedding
        embedding = bi_encoder.encode([question])[0]

        # Converter o embedding para uma string adequada para o PostgreSQL
        embedding_str = np.array2string(embedding, separator=',', precision=8, suppress_small=True)

        # Inserir no banco de dados
        cur.execute(
            "INSERT INTO physics_questions (question, embedding) VALUES (%s, %s)",
            (question, embedding_str)
        )

        conn.commit()
    except (psycopg2.DatabaseError, Exception) as error:
        print(f"Erro ao inserir questão no banco de dados: {error}")
        raise error  # Ou você pode escolher lidar de outra maneira
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

# Chame a função para inserir uma nova questão
#insert_question("Qual é a lei de Newton para o movimento de um objeto?")



def is_table_empty(cursor):
    
    cursor.execute("SELECT COUNT(*) FROM physics_questions")
    count = cursor.fetchone()[0]
    return count == 0

def check_table_exists(cursor):
    cursor.execute("""
    SELECT EXISTS (
        SELECT 1
        FROM information_schema.tables 
        WHERE table_schema = 'public'
        AND table_name = 'physics_questions'
    )
    """)
    return cursor.fetchone()[0]






def insert_questions(questions):
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # Carregar o modelo SBERT
        bi_encoder = SentenceTransformer("api/models/SBERT")

        # Codificar as questões em embeddings em lote
        embeddings = bi_encoder.encode(questions)

        # Preparar os dados para inserção no banco de dados
        data_to_insert = []
        for question, embedding in zip(questions, embeddings):
            # Converter o embedding para uma string adequada para o PostgreSQL
            embedding_str = np.array2string(embedding, separator=',', precision=8, suppress_small=True)
            data_to_insert.append((question, embedding_str))

        # Inserir no banco de dados em lote
        insert_query = "INSERT INTO physics_questions (question, embedding) VALUES (%s, %s)"
        cur.executemany(insert_query, data_to_insert)

        conn.commit()
    except (psycopg2.DatabaseError, Exception) as error:
        print(f"Erro ao inserir questões no banco de dados: {error}")
        raise error  # Ou você pode escolher lidar de outra maneira
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()
