"""外部 Agent 的 HTTP 适配器。

被测 Agent 通常已经是独立服务（独立虚拟环境、独立进程、甚至独立依赖版本），
没法在 agent-eval 进程内直接 import。本模块把任意「POST 一个 JSON、返回文本
+ 工具轨迹」的 HTTP 服务，包装成 ``TrackedAgentExecutor`` 能直接调用的对象。

接入契约（见 ``app/agents/executor.py``）只有一条::

    result = agent.invoke(payload, config={"callbacks": [tracker]})

返回值是 str 或任意可解析对象即可，轨迹通过 callback handler 回传。

为什么主动回放回调
------------------
``TrackedAgentExecutor`` 靠 LangChain 的回调钩子采集耗时与工具轨迹。HTTP
适配器拿不到真实的 LLM 响应对象，所以这里手工触发等价事件：

* ``on_llm_start`` / ``on_llm_end``：撑起端到端耗时（``total_duration``）；
* ``on_tool_start`` / ``on_tool_end``：把服务端返回的工具轨迹补进执行记录。

Token 用量：默认服务端不返回，恒为 0，成本维度因此不可用；若响应 JSON 里
带有 token 相关字段会自动识别并利用，不需要改本文件。
"""

import json
import logging
from typing import Any, Dict, List, Optional, Sequence
from uuid import uuid4

from app.monitoring.token_estimator import estimate_usage, usage_is_meaningful

logger = logging.getLogger(__name__)

# 服务端可能用来表示 Token 用量的字段名（按优先级尝试）
_USAGE_KEYS = ("token_usage", "usage_metadata", "usage", "tokens")
_PROMPT_KEYS = ("prompt_tokens", "input_tokens", "prompt")
_COMPLETION_KEYS = ("completion_tokens", "output_tokens", "completion")
_TOTAL_KEYS = ("total_tokens", "total")


# 服务端自测耗时字段名（只认带 ms 后缀的，避免把「秒」当成毫秒）
_LATENCY_KEYS = ("latency_ms", "duration_ms", "elapsed_ms", "took_ms", "response_time_ms")


def _extract_latency_ms(data: Dict[str, Any]) -> Optional[float]:
    """取服务端自己测的处理耗时（毫秒）。

    客户端计时里混着网络往返和请求排队，被测 Agent 自己报的才是真实处理时间。
    拿不到就返回 None，由 tracker 回退到客户端计时。
    """
    for key in _LATENCY_KEYS:
        value = data.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and value > 0:
            return float(value)
    return None


def _pick(raw: Dict[str, Any], names: Sequence[str]) -> int:
    for name in names:
        value = raw.get(name)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return int(value)
    return 0


def _extract_usage(data: Dict[str, Any]) -> Optional[Dict[str, int]]:
    """从响应 JSON 里尽力提取 Token 用量；没有则返回 None。"""
    for key in _USAGE_KEYS:
        raw = data.get(key)
        if not isinstance(raw, dict) or not raw:
            continue
        prompt = _pick(raw, _PROMPT_KEYS)
        completion = _pick(raw, _COMPLETION_KEYS)
        total = _pick(raw, _TOTAL_KEYS) or (prompt + completion)
        if prompt or completion or total:
            return {
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": total,
            }
    return None


class _SyntheticLLMResponse:
    """给 ``AgentExecutionTracker.on_llm_end`` 使用的最小响应对象。

    只需要 ``content``，以及空的 usage 结构（真实用量拿不到时保持为 0）。

    两个额外通道也能透传给 tracker：

    * ``token_source``：``"remote"``（服务端上报）或 ``"estimated"``（本地估算），
      决定性能维度敢不敢把这份用量当真；
    * ``latency_ms``：服务端自测耗时，比客户端计时更贴近 Agent 真实处理时间。
    """

    def __init__(self, content: str, usage: Optional[Dict[str, int]] = None,
                 token_source: Optional[str] = None,
                 latency_ms: Optional[float] = None) -> None:
        self.content = content
        self.response_metadata = {"token_usage": usage} if usage else {}
        if token_source:
            self.response_metadata["token_source"] = token_source
        if latency_ms:
            self.response_metadata["latency_ms"] = latency_ms
        self.usage_metadata: Dict[str, int] = {}
        self.llm_output: Optional[Dict[str, Any]] = None
        self.generations: List[Any] = []


class RemoteHTTPAgent:
    """把远端 HTTP 对话服务包装成本地 Agent。"""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        chat_path: str = "/api/v1/diagnosis/chat",
        request_field: str = "message",
        extra_payload: Optional[Dict[str, Any]] = None,
        reply_field: str = "reply",
        tool_calls_field: str = "tool_calls",
        model_name: str = "unknown",
        timeout: float = 300.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.url = f"{self.base_url}/{chat_path.lstrip('/')}"
        self.request_field = request_field
        self.extra_payload = dict(extra_payload or {})
        self.reply_field = reply_field
        self.tool_calls_field = tool_calls_field
        self.timeout = timeout
        self._model_name = model_name or "unknown"

    # ------------------------------------------------------------------ 契约
    @property
    def model_name(self) -> str:
        """供成本计算使用（``TrackedAgentExecutor`` 会读取该属性）。"""
        return self._model_name

    def invoke(self, payload: Any, config: Optional[Dict[str, Any]] = None) -> str:
        """执行一次远端调用，并把轨迹回传给 callback handler。"""
        prompt = self._extract_input(payload)
        handlers = self._handlers(config)

        body: Dict[str, Any] = {self.request_field: prompt, **self.extra_payload}

        self._notify(handlers, "on_llm_start", {"name": "remote_http_agent"}, [prompt])
        try:
            data = self._post(body)
        except Exception as exc:  # noqa: BLE001 - 交由上层按用例失败处理
            self._notify(handlers, "on_llm_error", exc)
            raise

        reply = str(data.get(self.reply_field) or "")
        tool_calls = data.get(self.tool_calls_field) or []
        self._replay_tool_calls(tool_calls, handlers)

        # 服务端上报优先；没有就按可见文本估算，并标记为 estimated
        usage = _extract_usage(data)
        token_source = None
        if usage_is_meaningful(usage):
            token_source = "remote"
        else:
            usage = estimate_usage(prompt, reply, tool_calls)
            token_source = "estimated" if usage_is_meaningful(usage) else None

        self._notify(
            handlers,
            "on_llm_end",
            _SyntheticLLMResponse(reply, usage, token_source, _extract_latency_ms(data)),
        )

        if not reply:
            logger.warning("远端 %s 返回空回复: %s", self.url, str(data)[:200])
        return reply

    # ------------------------------------------------------------------ 内部
    def _post(self, body: Dict[str, Any]) -> Dict[str, Any]:
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - 依赖缺失时给出明确提示
            raise RuntimeError("缺少 httpx，无法使用 remote_http 适配器") from exc

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(self.url, json=body)
            response.raise_for_status()
            payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError(f"远端返回不是 JSON 对象: {type(payload).__name__}")
        return payload

    @staticmethod
    def _extract_input(payload: Any) -> str:
        if isinstance(payload, dict):
            for key in ("input", "input_prompt", "question", "prompt", "message"):
                if key in payload:
                    return str(payload[key])
            return str(payload)
        return str(payload)

    @staticmethod
    def _handlers(config: Optional[Dict[str, Any]]) -> List[Any]:
        if not config:
            return []
        handlers = config.get("callbacks") or []
        return list(handlers) if isinstance(handlers, (list, tuple)) else [handlers]

    def _notify(self, handlers: List[Any], method: str, *args: Any) -> None:
        for handler in handlers:
            func = getattr(handler, method, None)
            if func is None:
                continue
            try:
                func(*args, run_id=uuid4())
            except Exception as exc:  # noqa: BLE001 - 回调异常不应中断调用
                logger.debug("回调 %s 执行失败: %s", method, exc)

    def _replay_tool_calls(self, calls: Any, handlers: List[Any]) -> None:
        """把服务端返回的工具轨迹补录成 on_tool_start / on_tool_end 事件。"""
        if not isinstance(calls, (list, tuple)):
            return

        for call in calls:
            if not isinstance(call, dict):
                continue
            name = str(call.get("tool") or call.get("name") or "unknown")
            args = call.get("input", call.get("args"))
            output = call.get("output", "")
            input_repr = (
                args if isinstance(args, str)
                else json.dumps(args, ensure_ascii=False, default=str)
            )

            self._notify(handlers, "on_tool_start", {"name": name}, input_repr)
            self._notify(
                handlers,
                "on_tool_end",
                output if isinstance(output, str) else json.dumps(output, ensure_ascii=False, default=str),
            )


def build_remote_agent(config: Dict[str, Any], default_model: str = "unknown") -> RemoteHTTPAgent:
    """按 agent_config 构造远端适配器。

    支持的字段（均有默认值，最简只需给 ``type`` 与 ``base_url``）::

        {
          "type": "remote_http",
          "base_url": "http://127.0.0.1:8000",
          "chat_path": "/api/v1/diagnosis/chat",
          "request_field": "message",
          "reply_field": "reply",
          "tool_calls_field": "tool_calls",
          "extra_payload": {"user_id": "eval"},
          "model": "glm-4.5-air",
          "timeout": 300
        }
    """
    extra_payload = config.get("extra_payload")
    if extra_payload is None:
        extra_payload = {"user_id": config.get("user_id", "eval")}

    return RemoteHTTPAgent(
        base_url=str(config.get("base_url") or config.get("endpoint") or "http://127.0.0.1:8000"),
        chat_path=str(config.get("chat_path") or "/api/v1/diagnosis/chat"),
        request_field=str(config.get("request_field") or "message"),
        extra_payload=dict(extra_payload),
        reply_field=str(config.get("reply_field") or "reply"),
        tool_calls_field=str(config.get("tool_calls_field") or "tool_calls"),
        model_name=str(config.get("model") or default_model),
        timeout=float(config.get("timeout", 300)),
    )
