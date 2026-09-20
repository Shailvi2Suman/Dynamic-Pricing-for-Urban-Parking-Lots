# Small, reproducible image that runs the pricing pipeline anywhere.
# Multi-stage-free on purpose — this is a batch job, not a service, so simple wins.

FROM python:3.11-slim

# Don't write .pyc files; stream logs straight out (better for container logs).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install deps first so this layer is cached until requirements.txt changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Then copy the source.
COPY . .

# Default command: run the end-to-end batch demo and write artefacts to /app/data.
CMD ["python", "run_batch.py", "--days", "14"]
