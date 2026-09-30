# Mem0 server (codigo oficial v2.2.1, contexto "upstream") + patch local: Ollama + Qdrant
FROM python:3.12-slim

WORKDIR /app

COPY --from=upstream requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir "mem0ai==2.2.1" ollama qdrant-client fastembed spacy \
 && python -m spacy download en_core_web_sm

COPY --from=upstream . .
COPY local_config.py patch_main.py ./
RUN python patch_main.py /app/main.py

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

CMD ["sh", "-c", "alembic upgrade head && uvicorn main:app --host 127.0.0.1 --port 8888"]
