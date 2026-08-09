FROM python:3.11-slim

WORKDIR /app

# Copy requirements first — Docker caches this layer
COPY requirements.txt .

# Install PyTorch CPU version first (avoids CUDA bloat)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Railway sets PORT env variable automatically
EXPOSE 8000

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]