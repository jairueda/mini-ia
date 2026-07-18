"""
main.py - Backend de la Mini IA (RAG + Llama vía Groq, gratis)
================================================================
Expone una API que tu frontend en HTML puede consultar desde
cualquier lugar (ej. tiiny.host).

Variables de entorno necesarias:
    GROQ_API_KEY -> obténla gratis en https://console.groq.com

Endpoints:
    POST /preguntar   { "pregunta": "..." } -> { "respuesta": "...", "fuentes": [...] }
    GET  /salud       -> chequeo de que el servidor está vivo
"""

import os
import glob
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
from pypdf import PdfReader
from pdf2image import convert_from_path
import pytesseract

# --------------------------------------------------------------
# Configuración
# --------------------------------------------------------------
DOCS_DIR = "documentos"
DB_DIR = "vectordb"
COLLECTION_NAME = "conocimiento"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
N_FRAGMENTOS = 4
MODELO = "llama-3.1-8b-instant"  # modelo gratuito de Groq

PROMPT_SISTEMA = """Eres un asistente experto que responde preguntas ÚNICAMENTE
usando la información de CONTEXTO que se te entrega. Si la respuesta no está
en el contexto, dilo claramente, no inventes datos. Responde en español,
de forma clara y breve."""

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

app = FastAPI(title="Mini IA API")

# Permite que tu HTML (en cualquier dominio, ej. tiiny.host) llame a esta API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------
# Ingesta de documentos (se corre automáticamente al iniciar
# el servidor si la base vectorial no existe todavía)
# --------------------------------------------------------------
def leer_txt(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def leer_pdf(path):
    """Extrae texto de un PDF. Si el PDF es escaneado (no tiene texto
    real, solo imágenes), hace OCR automáticamente página por página."""
    reader = PdfReader(path)
    texto = "\n".join(page.extract_text() or "" for page in reader.pages)

    # Si el texto extraído es muy corto, asumimos que es un PDF
    # escaneado y aplicamos OCR sobre cada página convertida a imagen.
    if len(texto.strip()) < 30:
        print(f"'{path}' parece escaneado, aplicando OCR...")
        paginas = convert_from_path(path, dpi=200)
        texto_ocr = []
        for i, pagina in enumerate(paginas):
            texto_pagina = pytesseract.image_to_string(pagina, lang="spa")
            texto_ocr.append(texto_pagina)
            print(f"  OCR página {i + 1}/{len(paginas)} lista")
        texto = "\n".join(texto_ocr)

    return texto


def dividir_en_chunks(texto, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    texto = texto.replace("\n", " ").strip()
    chunks, inicio = [], 0
    while inicio < len(texto):
        chunks.append(texto[inicio:inicio + chunk_size])
        inicio += chunk_size - overlap
    return [c.strip() for c in chunks if c.strip()]


def construir_base_si_no_existe():
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    client = chromadb.PersistentClient(path=DB_DIR)

    try:
        return client.get_collection(COLLECTION_NAME, embedding_function=embedding_fn)
    except Exception:
        pass  # no existe aún, la creamos abajo

    rutas = glob.glob(os.path.join(DOCS_DIR, "*.txt")) + \
            glob.glob(os.path.join(DOCS_DIR, "*.pdf"))

    documentos = []
    for ruta in rutas:
        texto = leer_pdf(ruta) if ruta.endswith(".pdf") else leer_txt(ruta)
        for i, chunk in enumerate(dividir_en_chunks(texto)):
            documentos.append({
                "id": f"{os.path.basename(ruta)}_{i}",
                "texto": chunk,
                "fuente": os.path.basename(ruta),
            })

    coleccion = client.create_collection(COLLECTION_NAME, embedding_function=embedding_fn)
    if documentos:
        coleccion.add(
            ids=[d["id"] for d in documentos],
            documents=[d["texto"] for d in documentos],
            metadatas=[{"fuente": d["fuente"]} for d in documentos],
        )
    return coleccion


coleccion = construir_base_si_no_existe()


# --------------------------------------------------------------
# API
# --------------------------------------------------------------
class PreguntaRequest(BaseModel):
    pregunta: str


@app.get("/salud")
def salud():
    return {"status": "ok"}


@app.post("/preguntar")
def preguntar(req: PreguntaRequest):
    resultados = coleccion.query(query_texts=[req.pregunta], n_results=N_FRAGMENTOS)
    fragmentos = resultados["documents"][0]
    fuentes = list(set(m["fuente"] for m in resultados["metadatas"][0]))

    if not fragmentos:
        return {
            "respuesta": "No tengo información cargada todavía sobre este tema.",
            "fuentes": [],
        }

    contexto_texto = "\n\n---\n\n".join(fragmentos)

    completion = groq_client.chat.completions.create(
        model=MODELO,
        messages=[
            {"role": "system", "content": PROMPT_SISTEMA},
            {"role": "user", "content": f"CONTEXTO:\n{contexto_texto}\n\nPREGUNTA: {req.pregunta}"},
        ],
    )

    return {
        "respuesta": completion.choices[0].message.content,
        "fuentes": fuentes,
    }
