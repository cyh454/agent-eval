"""集中式配置 (.env -> Settings)，替代原先散落的 os.getenv 读取方式。"""

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用配置。环境变量名与字段名同名，读取不区分大小写。"""

    model_config = SettingsConfigDict(
        env_file=(".env",),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---------- 应用 ----------
    APP_NAME: str = "agent-eval"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False
    API_PREFIX: str = ""

    # ---------- 数据库 ----------
    DATABASE_URL: str = "postgresql+psycopg2://postgres:postgres@localhost:5432/agent_eval"
    AUTO_CREATE_SCHEMA: bool = True
    SQL_ECHO: bool = False

    # ---------- 被测 / 裁判 LLM ----------
    LLM_MODEL: str = "glm-4.5-air"
    LLM_API_KEY: str = ""
    LLM_BASE_URL: Optional[str] = None
    LLM_TEMPERATURE: float = 0.0
    LLM_TIMEOUT: float = 60.0
    LLM_MAX_RETRIES: int = 2

    # 兼容历史变量名 OPENAI_API_KEY
    OPENAI_API_KEY: Optional[str] = None

    # ---------- Embedding ----------
    EMBEDDING_PROVIDER: str = "openai_compatible"
    EMBEDDING_MODEL: str = "embedding-3"
    EMBEDDING_DIMENSION: int = 1024
    EMBEDDING_API_KEY: Optional[str] = None
    EMBEDDING_BASE_URL: Optional[str] = None

    # ---------- 评估阈值 ----------
    DEFAULT_LATENCY_THRESHOLD: float = 5.0
    DEFAULT_COST_THRESHOLD: float = 0.1

    # ---------- 综合评分权重 ----------
    WEIGHT_QUALITY: float = 0.3
    WEIGHT_PERFORMANCE: float = 0.2
    WEIGHT_SECURITY: float = 0.25
    WEIGHT_TASK_COMPLETION: float = 0.25

    @property
    def llm_api_key(self) -> str:
        """优先 LLM_API_KEY，回退到历史上使用的 OPENAI_API_KEY。"""
        return self.LLM_API_KEY or self.OPENAI_API_KEY or ""

    @property
    def embedding_api_key(self) -> str:
        return self.EMBEDDING_API_KEY or self.llm_api_key

    @property
    def embedding_base_url(self) -> Optional[str]:
        return self.EMBEDDING_BASE_URL or self.LLM_BASE_URL


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """进程内共享同一份配置实例。"""
    return Settings()


def reload_settings() -> Settings:
    """测试场景：丢弃缓存重新读取环境变量。"""
    get_settings.cache_clear()
    return get_settings()
