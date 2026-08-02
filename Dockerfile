# 單一 image 給所有服務用（爬蟲 / ETL / 嵌入 / API / 儀表板）。
# 分兩個 image 沒有好處：同一台機器，layer 共用，實際磁碟成本是 0。
# 依賴全部由 uv 依 uv.lock 還原 —— 本機與容器裝到的是同一組版本。
FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.12.1 /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/data/hf_cache \
    TZ=Asia/Taipei

WORKDIR /app

# libgomp1: umap-learn / numba 需要；curl: healthcheck 用
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# 只複製鎖檔先裝依賴，改程式碼不會讓這層失效
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

COPY README.md ./
COPY jobshift/ ./jobshift/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

CMD ["python", "-c", "print('jobshift image ready；用 docker compose run 指定要跑的階段。')"]
