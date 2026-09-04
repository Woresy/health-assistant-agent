"""食物健康助手唯一启动入口。"""

from __future__ import annotations

import os

from src.ui.app import (
    APP_CSS,
    APP_HEAD,
    APP_THEME,
    demo,
)


if __name__ == "__main__":
    demo.launch(
        server_name=os.getenv("APP_HOST", "127.0.0.1"),
        server_port=int(os.getenv("APP_PORT", "7860")),
        show_error=True,
        css=APP_CSS,
        theme=APP_THEME,
        head=APP_HEAD,
    )
