FROM python:3.11-slim

# Prevent Python from writing pyc files and keeping stdout/stderr buffered
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create a non-root user
RUN useradd -m -s /bin/bash app && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
USER app

# Install Python packages
COPY --chown=app:app requirements.txt ./
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy application code
COPY --chown=app:app . .

# Add local user bin to PATH
ENV PATH="/home/app/.local/bin:${PATH}"

# Create necessary directories
RUN mkdir -p models results/training results/demo

# Command to run the application
CMD ["uvicorn", "src.web.app:app", "--host", "0.0.0.0", "--port", "8000"]
