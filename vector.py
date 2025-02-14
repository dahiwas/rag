from langchain.vectorstores.pgvector import PGVector
from pgvector.psycopg2 import register_vector
import psycopg2
import uuid
import json

from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
from credentials import DBNAME, USER, PASSWORD, HOST, PORT
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


def extrair_dados_texto(path_file):
    # Ler o PDF
    reader = PdfReader(path_file)
    documents = "\n".join(page.extract_text() for page in reader.pages)

    # Criar um objeto Document para LangChain
    document_obj = [Document(page_content=documents)]  # Criamos uma lista com um único Document

    # Dividir o texto em partes menores
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=450, chunk_overlap=20)
    texts = text_splitter.split_documents(document_obj)


    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    doc_vectors = model.encode([t.page_content for t in texts])
    
    print('Texts:')
    print(texts)
    
    print('Vetores')
    print(doc_vectors)
    
    return texts, doc_vectors


def setup_database_and_insert_embeddings(dbname, user, password, host, port, collection_name, metadata, embeddings, texts):
    conn = psycopg2.connect(dbname=dbname, user=user, password=password, host=host, port=port)
    cur = conn.cursor()

    cur.execute('CREATE EXTENSION IF NOT EXISTS vector')

    create_collections_table_query = """
    CREATE TABLE IF NOT EXISTS rslt_collections (
        id UUID PRIMARY KEY,
        collection_name TEXT NOT NULL,
        metadata JSONB
    );
    """
    create_embeddings_table_query = """
    CREATE TABLE IF NOT EXISTS rslt_embeddings (
        id UUID PRIMARY KEY,
        collection_id UUID,
        embedding vector,
        document TEXT,
        FOREIGN KEY (collection_id) REFERENCES rslt_collections (id)
    );
    """
    cur.execute(create_collections_table_query)
    cur.execute(create_embeddings_table_query)
    
    collection_id = uuid.uuid4()
    insert_collection_query = """
    INSERT INTO rslt_collections (id, collection_name, metadata)
    VALUES (%s, %s, %s);
    """
    cur.execute(insert_collection_query, (str(collection_id), collection_name, json.dumps(metadata)))
    
    insert_query = "INSERT INTO rslt_embeddings (id, collection_id, embedding, document) VALUES (%s, %s, %s, %s);"
    for i, (embedding, text) in enumerate(zip(embeddings, texts)):
        embedding_id = uuid.uuid4()
        cur.execute(insert_query, (str(embedding_id), str(collection_id), embedding.tolist(), str(text.page_content)))
        print(f"document {i} de {len(texts)}")
    
    conn.commit()
    cur.close()
    conn.close()
    
    print("Embeddings inseridos com sucesso.")
    return collection_id


path_file = "nihms-1572778.pdf"
texts, doc_vectors = extrair_dados_texto(path_file)

# Uso da função:
COLLECTION_NAME = "pm_remarks_vectors"
cid = setup_database_and_insert_embeddings(
    dbname=DBNAME,
    user=USER,
    password=PASSWORD,
    host=HOST,
    port=PORT,
    collection_name=COLLECTION_NAME,
    metadata={'creator': 'Desscrição de arquivos', 'description': 'Observações'},
    embeddings=doc_vectors,
    texts=texts
)
