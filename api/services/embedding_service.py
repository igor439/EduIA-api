import psycopg2
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from services.config import get_db_connection
from services.question_service import check_table_exists, insert_questions, is_table_empty


# Carregar o modelo de Cross-Encoder para refinamento da ordenação
cross_encoder = CrossEncoder("api/models/finetuned_SBERT", num_labels=1)


# Carregar o modelo de embeddings (SBERT)
bi_encoder = SentenceTransformer("api/models/SBERT")





def get_similar_questions_from_db(input_question):
    # Embeddings para a nova pergunta
    new_embedding = bi_encoder.encode([input_question], convert_to_tensor=False)
    faiss.normalize_L2(new_embedding)

    questions, embeddings_normalized = retrieve_questions_and_embeddings()

    # Criar índice FAISS
    dimension = embeddings_normalized.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Similaridade por produto interno
    index.add(embeddings_normalized)

    # Buscar as top 5 questões semelhantes usando FAISS
    k = 5
    distances, indices = index.search(new_embedding, k)

    # Preparar as top 5 questões semelhantes
    similar_questions = [questions[idx] for idx in indices[0]]

    # Criar pares de sentenças para o Cross-Encoder
    pairs = [[input_question, question] for question in similar_questions]

    # Gerar a similaridade entre os pares de sentenças
    similarity_scores = cross_encoder.predict(pairs)
    pairs_with_scores = list(zip(similar_questions, similarity_scores))
    pairs_with_scores.sort(key=lambda x: x[1], reverse=True)
    return pairs_with_scores



# Função para conectar ao banco de dados e recuperar questões e embeddings
def retrieve_questions_and_embeddings():
    # Configurações do banco de dados
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Recuperar questões e embeddings do banco de dados
    cur.execute("SELECT question, embedding FROM physics_questions")
    rows = cur.fetchall()
    
    questions = []
    embeddings = []
    
    for row in rows:
        question = row[0]
        embedding_str = row[1]
        embedding = np.fromstring(embedding_str.strip("[]"), sep=',')
        questions.append(question)
        embeddings.append(embedding)
    
    cur.close()
    conn.close()
    
    # Converter embeddings para numpy array e normalizar
    embeddings = np.array(embeddings)
    
    return questions, embeddings



def create_table():
    try:
        # Conectar ao banco de dados
        conn = get_db_connection()
        cur = conn.cursor()

        # Executar o comando de criação da tabela
        cur.execute("""
            CREATE TABLE IF NOT EXISTS physics_questions (
                id SERIAL PRIMARY KEY,
                question TEXT NOT NULL,
                embedding VECTOR(384) -- Para armazenar o vetor de embedding
            )
        """)
        
        # Confirmar a transação
        conn.commit()

    except (psycopg2.DatabaseError, Exception) as error:
        print(f"Erro ao criar a tabela: {error}")
        raise  # Relevantar o erro para ser tratado posteriormente se necessário
    finally:
        # Fechar cursor e conexão
        if cur:
            cur.close()
        if conn:
            conn.close()


def init_db():


    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        if not check_table_exists(cur):
            create_table()
            print("Tabela criada com sucesso.")

        if is_table_empty(cur):
            questoes = ["Um projétil é disparado com velocidade inicial de 50 m/s em um ângulo de 30 graus com a horizontal. Qual é a altura máxima atingida pelo projétil?",
                        "Calcule a altura máxima de um projétil lançado a 50 m/s com um ângulo de 30° em relação ao solo. a)31,25m b) 45,75m c) 19,08m d) 25,00m",
                        "Um bloco de 5 kg é puxado com uma força de 20 N em uma superfície horizontal sem atrito. Qual é a aceleração do bloco?",
                        "Determine a aceleração de um bloco de 5 kg sob uma força de 20 N em um plano horizontal sem atrito. a) 5 m/s² b) 4 m/s² c) 3 m/s²  d) 6 m/s²",
                        "Um carro de 1500kg faz uma curva de raio 50 m a uma velocidade constante de 15 m/s. Qual é a força centrípeta necessária para manter o carro na curva?",
                        "Calcule a força centrípeta necessária para um carro de 1500 kg fazer uma curva de raio 50 m a 15 m/s a)4500 N b)6750 N c)9000 N d)11250 N",
                        "Um trabalhador empurra uma caixa de 20 kg por 10 m com uma força constante de 50 N. Qual é o trabalho realizado pelo trabalhador?",
                        "Calcule o trabalho realizado ao empurrar uma caixa de 20 kg por 10 m com uma força de 50 N. a)300 J b)400 J c)500 J d)600 J",
                        "Um cilindro sólido de massa 4 kg e raio 0,3 m gira a 8 rad/s. Qual é a energia cinética rotacional do cilindro?",
                        "Calcule a energia cinética rotacional de um cilindro de 4 kg e raio 0,3 m girando a 8 rad/s. a)3,84 J b)4,48 J c)5,12 J d)6,48 J",
                    ]
            insert_questions(questoes)
            print("Questões inseridas com sucesso.")
        else:
            print("A tabela já contém dados. Nenhuma inserção realizada.")
        
    except Exception as e:
        print(f"Erro ao configurar o DB: {e}")
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()