FROM python:3.10-slim-bullseye

RUN apt-get update && \
    apt-get install -y libcairo2-dev pkg-config build-essential ffmpeg libsm6 libxext6 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD [ "streamlit", "run", "RAG-v2-gemini.py" ]