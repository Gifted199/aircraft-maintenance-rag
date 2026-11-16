# syntax=docker/dockerfile:1
#
# The Streamlit UI, as its own small image (separate from the API image):
#   docker build --platform linux/amd64 -t akinsds1126-aero-rag-ui:0.2.0 ui
#
# The build context is the ui/ folder only, so the image cannot see .env, src/ or the spec. It
# contains Streamlit and two Python files; it holds no secrets and talks only to API_URL.
FROM --platform=linux/amd64 python:3.12-slim

# A dedicated unprivileged user. Streamlit writes a small config folder in its home directory.
RUN useradd --system --uid 10001 --user-group --create-home --home-dir /home/app app

WORKDIR /app

# Dependencies first (cached until requirements.txt changes). requirements.txt is a fully pinned
# lock file compiled from requirements.in, so every build gets the same versions.
COPY requirements.txt ./
RUN pip install --no-cache-dir --disable-pip-version-check -r requirements.txt

COPY app.py presenter.py ./

# API_URL: on ECS the UI and API containers share the task's network, so the API is localhost.
# Streamlit settings: listen on all interfaces, no browser, no usage statistics, no file watcher
# (no live reload in production), and the toolbar in viewer mode (no developer menu).
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    API_URL=http://localhost:8000 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_CLIENT_TOOLBAR_MODE=viewer

USER app
EXPOSE 8501

# Streamlit's own health endpoint. python is used because slim has no curl.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)"]

CMD ["streamlit", "run", "app.py"]
