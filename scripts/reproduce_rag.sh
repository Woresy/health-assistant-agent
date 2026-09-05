#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${RAG_VENV_DIR:-${PROJECT_ROOT}/.venv}"
BOOTSTRAP_PYTHON="${RAG_PYTHON_BIN:-python3}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    echo "[setup] 创建 Python 虚拟环境：${VENV_DIR}"
    if ! "${BOOTSTRAP_PYTHON}" -m venv "${VENV_DIR}"; then
        echo "错误：无法创建虚拟环境。请安装 Python 3.11+ 和 venv 模块。" >&2
        exit 2
    fi
fi

if ! "${VENV_DIR}/bin/python" -c \
    'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)'; then
    echo "错误：RAG 复现需要 Python 3.11+，当前虚拟环境版本过低。" >&2
    echo "请删除或更换 RAG_VENV_DIR，并通过 RAG_PYTHON_BIN 指定 Python 3.11+。" >&2
    exit 2
fi

if [[ "${RAG_SKIP_INSTALL:-0}" != "1" ]]; then
    echo "[setup] 安装 requirements.txt 中的固定依赖"
    "${VENV_DIR}/bin/python" -m pip install \
        --disable-pip-version-check \
        -r "${PROJECT_ROOT}/requirements.txt"
fi

cd "${PROJECT_ROOT}"
exec "${VENV_DIR}/bin/python" scripts/reproduce_rag.py "$@"
