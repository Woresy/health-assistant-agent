FROM python:3.11.16-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN addgroup --system --gid 10001 healthos \
    && adduser --system --uid 10001 --ingroup healthos --home /home/healthos healthos

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY app.py ./
COPY src ./src
COPY data ./data
COPY scripts/backup_sqlite.py ./scripts/backup_sqlite.py

RUN mkdir -p /app/runtime \
    && chown -R healthos:healthos /app/runtime

USER healthos

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/', timeout=4)"

CMD ["python", "app.py"]
