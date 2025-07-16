# Dockerfile

# 1. Base Image: Use an official Python slim image for a smaller footprint.
# Make sure the Python version matches what you've been using locally (e.g., 3.10, 3.11).
FROM python:3.10-slim

# 2. Set Environment Variables
ENV PYTHONDONTWRITEBYTECODE 1  # Prevents python from writing .pyc files to disc
ENV PYTHONUNBUFFERED 1        # Prevents python from buffering stdout and stderr

# 3. Set Work Directory
WORKDIR /app

# 4. Install OS-level dependencies (if any)
# For PyMuPDF (Fitz), you might need some system libraries if not already in python:slim.
# Often, for PyMuPDF, no extra OS libs are needed with recent wheels, but good to be aware.
# RUN apt-get update && apt-get install -y --no-install-recommends some-lib if needed

# 5. Install Python Dependencies (CPU-only PyTorch for smaller image size)
# First, install PyTorch CPU version explicitly to ensure sentence-transformers uses it.
# Check PyTorch website for the latest CPU-only wheel URL if needed.
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Copy requirements.txt and install the rest of the dependencies.
# Ensure 'torch', 'torchvision', 'torchaudio' are NOT in requirements.txt
# OR if they are, they don't specify a version that conflicts with the CPU one above.
# Best practice: Remove them from requirements.txt if installing them separately like this.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy Application Code and Data
COPY ./main.py .


# CRITICAL: Copy the pre-populated ChromaDB data directory
# Ensure 'chroma_data' directory exists in your project root and is populated
# by running doc_ingestor.py locally before building the Docker image.
COPY ./chroma_data ./chroma_data

# Also copy the .env file for local Docker builds (will be overridden by Fly secrets in production)
# For Fly.io, you will set GROQ_API_KEY as a secret, not using the .env file in the image.
# However, it can be useful for local docker testing.
# Consider .dockerignore for .env if you don't want it in the image,
# but then you must provide env vars differently for local docker runs.
# For simplicity in this step, we'll copy it, assuming it only has GROQ_API_KEY.
# COPY .env .

# 7. Expose Port (the port Uvicorn will run on *inside* the container)
EXPOSE 8000

# 8. Define Command to Run Application
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
# For Fly.io, they recommend running gunicorn with uvicorn workers for production.
# Example: CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:8000"]
# Let's start with the simpler uvicorn command, Fly.io's default process might also pick up FastAPI well.
# Fly.io's default for Python often uses Gunicorn. If using `flyctl launch` it might set this up.
# For now, this is fine. `fly launch` might override or suggest a better CMD.
RUN which uvicorn
RUN uvicorn --version
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]