FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    espeak-ng \
    git \
    git-lfs \
    build-essential \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir google-cloud-storage

# Pre-download NLTK data
RUN python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords')"

# Download StyleTTS2-LJSpeech model weights
RUN git clone https://huggingface.co/yl4579/StyleTTS2-LJSpeech \
    && mv StyleTTS2-LJSpeech/Models Audio \
    && rm -rf StyleTTS2-LJSpeech

COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

ENV CUBLAS_WORKSPACE_CONFIG=:4096:8

ENTRYPOINT ["/app/entrypoint.sh"]
