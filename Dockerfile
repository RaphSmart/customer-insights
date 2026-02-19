# Base image: slim Python
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# System dependencies for PyTorch & ML
RUN apt-get update && apt-get install -y \
        build-essential \
        curl \
        git \
        && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install PyTorch + NumPy <2 + CPU packages
RUN pip install --no-cache-dir \
        torch==2.2.0+cpu \
        torchvision==0.17.0+cpu \
        torchaudio==2.2.0+cpu \
        "numpy<2" \
        -f https://download.pytorch.org/whl/cpu/torch_stable.html

# Copy requirements and install other dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the necessary source code
COPY src/ ./src
# COPY models/distilbert ./models/distilbert

# Expose API port
EXPOSE 8080

# Set default command
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
