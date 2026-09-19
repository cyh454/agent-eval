"""数据库连接层：引擎、Session 工厂与声明式基类。

所有 ORM 模型都继承此处的 ``Base``，通过 ``get_db`` 提供的 Session 依赖注入到路由。
"""

from typing import Any, Dict, Iterator

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import get_settings


class Base(DeclarativeBase):
    """所有 ORM 模型的公共基类。"""


def _build_kwargs(url: str) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {"pool_pre_ping": True}
    if url.startswith("sqlite"):
        # SQLite 允许跨线程复用连接（FastAPI 同步路由在线程池中执行）
        kwargs["connect_args"] = {"check_same_thread": False}
    return kwargs


def build_engine() -> Engine:
    settings = get_settings()
    return create_engine(
        settings.DATABASE_URL,
        echo=settings.SQL_ECHO,
        **_build_kwargs(settings.DATABASE_URL),
    )


engine: Engine = build_engine()

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
    class_=Session,
)


def get_db() -> Iterator[Session]:
    """FastAPI 依赖项：每个请求一个独立 Session，请求结束即关闭。"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def ensure_vector_extension(target_engine: Engine | None = None) -> None:
    """创建 pgvector 扩展。SQLite 等非 Postgres 环境静默跳过。"""
    target_engine = target_engine or engine
    if target_engine.dialect.name != "postgresql":
        return
    with target_engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))


def init_db(target_engine: Engine | None = None) -> None:
    """初始化数据库结构（仅用于本地开发 / 测试，生产请用 alembic 迁移）。"""
    from app.models import Evaluation, EvaluationMetric, SecurityLog, SecurityRule  # noqa: F401

    target_engine = target_engine or engine
    ensure_vector_extension(target_engine)
    Base.metadata.create_all(bind=target_engine)


def dispose_engine() -> None:
    """释放连接池，供测试或应用关闭时调用。"""
    engine.dispose()
