# Imagen base: Python 3.11 ligero
FROM python:3.11-slim

# Instala Tesseract (el motor de OCR) + el paquete de idioma español
# + poppler-utils (necesario para convertir PDF a imágenes)
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-spa \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD uvicorn main:app --host 0.0.0.0 --port $PORT
