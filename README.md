# 食物健康助手

一个通过图片入口、手动食物检索和确定性营养计算完成本地饮食记录的 Gradio 学习项目。

## 运行环境

- Python 3.11+
- Gradio
- Pydantic
- Pillow
- pytest

所有直接依赖均已固定版本。应用不接入大模型，不需要任何 API Key。

## 从空虚拟环境启动

进入仓库根目录后，复制执行：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py