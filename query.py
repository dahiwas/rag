from langchain.vectorstores.pgvector import PGVector
from pgvector.psycopg2 import register_vector
import psycopg2
import uuid
import json

from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
from PyPDF2 import PdfReader
from langchain.schema import Document

import os
from dotenv import load_doetenv

DBNAME = os.getenv("DBNAME")
USER = os.getenv("USER")
PASSWORD = os.getenv("PASSWORD")
HOST = os.getenv("HOST")
PORT = os.getenv("PORT")
OPENAI = os.getenv("OPENAI")


def perform_similarity_search(dbname, user, password, host, port, query_embedding, top_k=10, collection_id=None):
    conn = psycopg2.connect(dbname=dbname, user=user, password=password, host=host, port=port)
    register_vector(conn)
    cur = conn.cursor()
    cur.execute('CREATE EXTENSION IF NOT EXISTS vector')

    if collection_id is None:
        similarity_search_query = """
        SELECT document, (embedding <-> %(query_embedding)s) AS distance
        FROM rslt_embeddings
        ORDER BY distance
        LIMIT %(top_k)s;
        """
        params = {'query_embedding': query_embedding, 'top_k': top_k}
    else:
        similarity_search_query = """
        SELECT document, (embedding <-> %(query_embedding)s) AS distance
        FROM rslt_embeddings
        WHERE collection_id = %(collection_id)s
        ORDER BY distance
        LIMIT %(top_k)s;
        """
        params = {'query_embedding': query_embedding, 'top_k': top_k, 'collection_id': collection_id}

    cur.execute(similarity_search_query, params)
    results = cur.fetchall()
    cur.close()
    conn.close()
    return results


def query(pergunta):

# Exemplo de uso:
    model = SentenceTransformer("all-MiniLM-L6-v2")

    query_embedding = model.encode(pergunta)
    results = perform_similarity_search(DBNAME, USER, PASSWORD, HOST, PORT, query_embedding, top_k=10)
    #for document, distance in results:
        #print(document, distance)



    from openai import OpenAI

    client = OpenAI(api_key=OPENAI)

    template = f"""
    Use APENAS o contexto para responder a pergunta abaixo, nada alem disso!.
    Se não souber a resposta, seja claro que nao temos informações no nosso banco de dados. Não invente respostas, NUNCA, nao enrole a resposta tambem.
    Se o contexto não estiver claro sobre a pergunta, diga que não sabe.
    Sempre diga que o contexto é o seu banco de dados.
    Use no máximo sete frases e mantenha a resposta concisa.
    Sempre termine a resposta com "obrigado por perguntar! Para mais informações falar com a Nutróloga da DNA Rafaela Amaral, Whatsapp 16 98989898".

    Pergunta: {pergunta}

    Contexto: {results}
    """

    #print('este eh o template:', template)


    response = client.chat.completions.create(model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": template}
        ])

    final_response = response.choices[0].message.content
    print(final_response)

print('\n\n\n')
for i in range(100):
    pergunta = input("\nFaca uma pergunta\n")
    query(pergunta)

