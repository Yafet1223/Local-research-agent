FROM python:3.12-slim

WORKDIR /app

# Use host Docker internal address for Ollama when running in container.
ENV OLLAMA_BASE_URL=http://host.docker.internal:11434
ENV OLLAMA_MODEL=qwen2.5:0.5b

# Install dependencies
COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose backend port
EXPOSE 5000

# Start FastAPI app
CMD ["uvicorn", "Backend.app:app", "--host", "0.0.0.0", "--port", "5000"]