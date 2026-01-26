FROM python:3.10-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ---- CRITICAL: CPU-only torch FIRST ----
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
    torch==2.1.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install app deps (torch already satisfied)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy ONLY inference code + model
COPY src/api ./src/api
COPY src/models ./src/models
COPY models/distilbert ./models/distilbert

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
