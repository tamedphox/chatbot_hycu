FROM python:3.11-slim

WORKDIR /app

# CPU-only runtime system packages
# ffmpeg/ffprobe: video STT processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# CPU PyTorch index is used for sentence-transformers / faster-whisper dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY res ./res

RUN mkdir -p /app/data/lectures /app/data/chroma_db /app/.hf_cache

ENV HF_HOME=/app/.hf_cache
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_PORT=8888
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV STREAMLIT_SERVER_FILE_WATCHER_TYPE=none

EXPOSE 8888

CMD ["streamlit", "run", "main.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8888", \
     "--server.fileWatcherType=none"]
