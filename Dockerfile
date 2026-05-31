FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000 8501
CMD ["sh", "-c", "if [ \"$SERVICE\" = \"api\" ]; then uvicorn src.api.main:app --host 0.0.0.0 --port 8000; else streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0; fi"]
# Render deploy fix
