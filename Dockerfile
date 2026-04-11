FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

COPY gsa_env/ /app/gsa_env/

ENV GSA_NUM_EPISODES=10

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "gsa_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
