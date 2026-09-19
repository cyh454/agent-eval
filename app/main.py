import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# 直接以脚本方式运行 app/main.py 时，sys.path[0] 是 app/ 目录而非项目根目录，
# 会导致 `import app` 失败。这里把项目根目录补进模块搜索路径。
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.api.datasets import router as datasets_router
from app.api.evaluations import router as evaluations_router
from app.api.security import router as security_router
from app.api.web import router as web_router
from app.core.config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
)
logger = logging.getLogger("agent-eval")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时按需建表。"""
    settings = get_settings()
    logger.info("初始化数据库结构...")

    if settings.AUTO_CREATE_SCHEMA:
        try:
            from app.database import init_db

            init_db()
        except Exception as exc:  # noqa: BLE001 - 数据库不可用时仍要提供接口，便于排查
            logger.warning("初始化数据库失败（服务继续启动）: %s", exc)
    else:
        logger.info("AUTO_CREATE_SCHEMA 已关闭，跳过建表（请通过 alembic 维护表结构）")

    logger.info("%s v%s 启动完成，接口文档: /docs",
                settings.APP_NAME, settings.APP_VERSION)

    # 由 __main__ 写入，直接运行文件时才会打印可点的网页地址
    ui_port = os.getenv("APP_PORT")
    if ui_port:
        logger.info("网页版一键评估: http://127.0.0.1:%s/ui", ui_port)

    yield

    logger.info("服务已停止")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="Agent 评估框架：质量 / 性能 / 安全 / 任务完成度四维度评估",
        debug=settings.DEBUG,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(datasets_router, prefix=settings.API_PREFIX)
    app.include_router(evaluations_router, prefix=settings.API_PREFIX)
    app.include_router(security_router, prefix=settings.API_PREFIX)
    app.include_router(web_router, prefix=settings.API_PREFIX)

    # 网页版一键评估（app/web/index.html）
    web_dir = Path(__file__).resolve().parent / "web"
    if web_dir.is_dir():
        app.mount("/ui", StaticFiles(directory=str(web_dir), html=True), name="ui")

    @app.get("/", include_in_schema=False)
    def index():
        return RedirectResponse(url="/ui/")

    @app.get("/health", tags=["system"], summary="健康检查")
    def health() -> dict:
        return {"status": "ok", "version": settings.APP_VERSION}

    @app.get("/metrics", tags=["system"], summary="Prometheus 指标")
    def metrics() -> Response:
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()


if __name__ == "__main__":
    # 直接运行本文件时启动开发服务器（等价于 uvicorn app.main:app --reload）。
    # 传 "app.main:app" 字符串而非 app 对象，reload 才能正常重载子进程。
    import socket

    import uvicorn

    # 8000 留给被测 Agent（如 vehicle-diagnosis-agent）：两者同占时本机进程胜出，
    # agent-eval 会误把自己的接口当成被测对象来评测。
    DEFAULT_PORT = 8010

    def _pick_port(start: int, tries: int = 12) -> int:
        """从 start 起找第一个空闲端口。

        默认起点已经避开被测服务的 8000，这里只是防止默认端口被别的实例占着时直接崩溃。
        """
        for offset in range(tries):
            port = start + offset
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                try:
                    sock.bind(("127.0.0.1", port))
                except OSError:
                    continue
                return port
        return start + tries

    port = _pick_port(int(os.getenv("PORT", str(DEFAULT_PORT))))
    # 传给 lifespan 打印可点地址，reload 子进程继承；自指检测也靠它识别自己
    os.environ["APP_PORT"] = str(port)

    uvicorn.run("app.main:app", host="127.0.0.1", port=port, reload=True)
