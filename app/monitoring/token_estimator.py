"""被测服务不回传 Token 用量时的估算器。

为什么要估
----------
多数外部 Agent 服务（含 vehicle-diagnosis-agent）的响应里只有文本和工具轨迹，
没有 usage 字段。拿不到 token 数的话，性能维度会失真：

* ``total_cost`` 恒为 0 → ``cost_score = 1 - 0/阈值`` 恒为**满分**（白白送掉 30% 权重）；
* ``token_efficiency`` 恒为 0（白白丢掉 20% 权重）。

结果是「性能分」只反映了快慢，却被当成四个子维度都测过了一样。

估的是什么
----------
只能估**外部可见的文本**：提问、最终回复、工具调用的输入与输出。
估不到的部分——被测 Agent 内部的系统提示词、RAG 召回的上下文、
中间轮的思考过程——恰恰是这类 Agent 消耗最大的地方。

所以对 RAG / 多工具编排的 Agent，这个数是**系统性低估的下界**：
它适合在同一批用例之间横向比较（谁更絮叨、谁调了更多工具），
**不能当作真实账单**，也不能和被测服务亲自上报的用量混为一谈。

因此凡是估算出来的值都必须带 ``token_source="estimated"`` 一路透传到
``AgentExecution.execution_metadata`` 与最终的评测报告，让读数的人知道它可信度几何。

编码选择
--------
统一用 ``cl100k_base``。它不是各家模型（GLM / Qwen 等）的真实分词表，
只是通用的近似口径——这里要的是可比的相对量，不是精确计费。
"""

import logging
import math
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_ENCODING: Any = None
_ENCODING_LOADED = False

# tiktoken 不可用时的粗估系数（中文一个字约一个 token，英文约 4 个字符一个）
_FALLBACK_RATIO = 0.6


def _get_encoding() -> Any:
    """惰性加载 tiktoken 编码器，失败则缓存 None 不再重试。"""
    global _ENCODING, _ENCODING_LOADED
    if _ENCODING_LOADED:
        return _ENCODING
    _ENCODING_LOADED = True
    try:
        import tiktoken

        _ENCODING = tiktoken.get_encoding("cl100k_base")
    except Exception as exc:  # noqa: BLE001 - 装不上也要能继续评测
        logger.info("tiktoken 不可用，改为按字符数粗估 token: %s", exc)
        _ENCODING = None
    return _ENCODING


def count_tokens(text: Any) -> int:
    """统计一段文本的近似 token 数，空文本返回 0。"""
    if text is None:
        return 0
    raw = text if isinstance(text, str) else str(text)
    if not raw.strip():
        return 0

    encoding = _get_encoding()
    if encoding is not None:
        try:
            return len(encoding.encode(raw))
        except Exception as exc:  # noqa: BLE001
            logger.debug("token 编码失败，回退粗估: %s", exc)

    # 中文占比高的文本按字数算，纯 ASCII 按 4 字符一个 token
    cjk = sum(1 for ch in raw if "一" <= ch <= "鿿")
    ascii_len = len(raw) - cjk
    rough = cjk + ascii_len / 4.0
    return max(1, int(math.ceil(rough * _FALLBACK_RATIO)))


def _tool_text(tool_call: Any) -> str:
    """把一条工具轨迹摊平成可计数的文本。"""
    if not isinstance(tool_call, dict):
        return str(tool_call)

    parts = []
    for key in ("tool", "name"):
        if tool_call.get(key):
            parts.append(str(tool_call[key]))
            break
    for key in ("input", "args", "output", "result"):
        value = tool_call.get(key)
        if value in (None, ""):
            continue
        parts.append(value if isinstance(value, str) else str(value))
    return " ".join(parts)


def estimate_usage(question: str, reply: str,
                   tool_calls: Optional[Any] = None) -> Dict[str, int]:
    """估算一次外部调用的 token 用量。

    口径：

    * ``completion_tokens`` = 最终回复；
    * ``prompt_tokens`` = 提问 + 工具轨迹（工具的输入输出会回填进下一轮的上下文，
      因此计入输入侧，而不是产出侧）；
    * ``total_tokens`` = 两者之和。

    返回全 0 表示确实没有可估算的文本，调用方应据此降级，而不是当成「消耗为 0」。
    """
    completion = count_tokens(reply)

    prompt_parts = [count_tokens(question)]
    if isinstance(tool_calls, (list, tuple)):
        prompt_parts.extend(count_tokens(_tool_text(call)) for call in tool_calls)

    prompt = sum(prompt_parts)
    total = prompt + completion

    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
    }


def usage_is_meaningful(usage: Optional[Dict[str, Any]]) -> bool:
    """判断一份用量数据是否「有观测意义」。

    全 0 等同于缺失——那是没采集到，不代表真的没消耗。
    """
    if not usage:
        return False
    return any(int(usage.get(key) or 0) > 0 for key in ("prompt_tokens", "completion_tokens", "total_tokens"))
