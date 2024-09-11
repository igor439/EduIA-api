import psycopg2

def get_db_connection():
    return psycopg2.connect(
        dbname="questions",
        user="postgres",
        password="40028922",
        host="localhost",
        port="5432"
    )