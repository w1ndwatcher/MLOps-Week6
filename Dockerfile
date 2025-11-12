# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Copy requirements (create requirements.txt)
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . /app

# Expose port (uvicorn default)
EXPOSE 8000

# Run uvicorn
CMD ["uvicorn", "iris_fastapi:app", "--host", "0.0.0.0", "--port", "8000"]