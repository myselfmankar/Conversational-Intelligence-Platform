# ==============================================================================
# STAGE 1: The "Builder" Stage
# This stage is responsible for installing dependencies and downloading models
# ==============================================================================
FROM python:3.12-slim AS builder

WORKDIR /app


RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*


RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY ./scripts/download_models.py .

RUN python download_models.py


# ==============================================================================
# STAGE 2: The Final "Runner" Stage
# This stage uses the pre-built components from the "Builder" stage
# ==============================================================================
FROM python:3.12-slim

WORKDIR /app

# --- Copy Pre-built Components from the "Builder" Stage ---

COPY --from=builder /opt/venv /opt/venv

COPY --from=builder /root/.cache /root/.cache

COPY . .


ENV PATH="/opt/venv/bin:$PATH"

# Set the Hugging Face cache directory to the location we copied our models to
# This explicitly tells the transformers library where to look.
ENV HF_HOME="/root/.cache/huggingface"

# Expose the port that Streamlit runs on
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableCORS=false"]