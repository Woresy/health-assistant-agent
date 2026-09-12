FROM python:3.11.16-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/huggingface

RUN addgroup --system --gid 10001 healthos \
    && adduser --system --uid 10001 --ingroup healthos --home /home/healthos healthos

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY app.py ./
COPY src ./src
COPY data ./data
COPY scripts/backup_sqlite.py ./scripts/backup_sqlite.py
COPY scripts/prepare_health_knowledge_model.py ./scripts/prepare_health_knowledge_model.py

# 文档向量随仓库提交，查询编码器不提交；镜像里没有它，健康知识 Dense 召回会
# 静默降级为词法检索。按索引 manifest 固定的模型和 revision 预置编码器，随后
# 锁成离线，运行时不再访问 Hugging Face。
RUN python scripts/prepare_health_knowledge_model.py \
    && chown -R healthos:healthos /opt/huggingface

ENV HF_HUB_OFFLINE=1

RUN mkdir -p /app/runtime \
    && chown -R healthos:healthos /app/runtime

USER healthos

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/', timeout=4)"

CMD ["python", "app.py"]
