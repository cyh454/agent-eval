"""LLM 工厂与 LLM 输出解析工具。

集中这里是为了解决原代码的几个问题：
1. 各评估器取模型名的 key 不统一（``model`` / ``llm_model``），配置项其实从未生效；
2. 没有传 ``base_url``，导致 OpenAI 兼容端点（如智谱 open.bigmodel.cn）无法使用；
3. LLM 返回被 ```` ```json ```` 代码块包裹时 json.loads 直接失败。
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

# 构造 ChatOpenAI 时若完全无 key 会直接抛错，导致整个服务起不来。
# 用占位符保证可以构造，真实调用失败时由调用方的 try/except 兜底。
_PLACEHOLDER_KEY = "missing-api-key"

_JSON_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.+?)```", re.DOTALL)
_FLOAT_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def build_chat_model(
    config: Optional[Dict[str, Any]] = None,
    settings: Optional[Settings] = None,
    temperature: Optional[float] = None,
    **overrides: Any,
) -> ChatOpenAI:
    """构建统一配置的 ChatOpenAI 实例。

    优先级：显式 overrides > config 字典 > .env 全局配置。
    """
    settings = settings or get_settings()
    config = config or {}

    # 允许直接传入已构造好的 LLM 实例（自定义实现或测试注入）
    injected = overrides.get("llm") or config.get("llm")
    if injected is not None:
        return injected

    model = (
        overrides.get("model")
        or config.get("model")
        or config.get("llm_model")
        or settings.LLM_MODEL
    )
    api_key = overrides.get("api_key") or config.get("api_key") or settings.llm_api_key
    base_url = (
        overrides.get("base_url") or config.get("base_url") or settings.LLM_BASE_URL
    )

    temp = temperature
    if temp is None:
        temp = config.get("temperature", settings.LLM_TEMPERATURE)

    if not api_key:
        logger.warning(
            "未配置 LLM_API_KEY，LLM 相关评估将返回兜底结果；请在 .env 中补充密钥"
        )

    kwargs: Dict[str, Any] = {
        "model": model,
        "temperature": temp,
        "api_key": api_key or _PLACEHOLDER_KEY,
        "timeout": settings.LLM_TIMEOUT,
        "max_retries": settings.LLM_MAX_RETRIES,
    }
    if base_url:
        # OpenAI 兼容端点必须显式指定，否则会打向 api.openai.com
        kwargs["base_url"] = base_url

    return ChatOpenAI(**kwargs)


def parse_json_response(text: str, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """从 LLM 输出中稳健地提取 JSON 对象。

    依次尝试：直接解析 -> ```json 代码块 -> 截取首尾花括号。
    """
    if default is None:
        default = {}

    if not text or not isinstance(text, str):
        return default

    candidates: List[str] = [text.strip()]

    fence = _JSON_FENCE_RE.search(text)
    if fence:
        candidates.append(fence.group(1).strip())

    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed
        return default

    return default


def parse_float_response(text: str, default: float = 0.0, lo: float = 0.0, hi: float = 1.0) -> float:
    """从 LLM 输出中提取一个数值并夹取到 [lo, hi] 区间。"""
    if not text or not isinstance(text, str):
        return default

    match = _FLOAT_RE.search(text)
    if not match:
        return default

    try:
        value = float(match.group(0))
    except ValueError:
        return default

    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def parse_bool_response(text: str, default: bool = False) -> bool:
    """从 LLM 输出中判定布尔值。"""
    if not text or not isinstance(text, str):
        return default

    normalized = text.strip().lower()
    true_words = ("true", "yes", "是", "成功", "完成")
    false_words = ("false", "no", "否", "失败", "未完成")

    if any(word in normalized for word in true_words):
        return True
    if any(word in normalized for word in false_words):
        return False

    return default
