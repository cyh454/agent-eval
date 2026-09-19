"""自动探测外部 Agent 服务的对话端点。

目标：用户只提供一个根地址（如 ``http://localhost:8000/``），就能自动推断出
``chat_path`` / ``request_field`` / ``reply_field`` / ``tool_calls_field``，
不必手工填字段映射。

做法：拉取目标服务的 OpenAPI 文档（FastAPI / Fastify / Springdoc 等都会生成），
在所有 POST 端点里挑一个"最像对话接口"的，再从它的请求 / 响应模型里按优先级
挑选字段。探测失败时回退到内置的默认值（贴合 vehicle-diagnosis-agent）。
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# 请求体里用来承载用户提问的字段名（按优先级）
PROMPT_FIELDS = ("message", "input", "query", "question", "prompt", "user_input", "text")
# 响应体里用来承载最终回复的字段名
REPLY_FIELDS = ("reply", "response", "output", "answer", "content", "result", "text")
# 响应体里承载工具调用轨迹的字段名
TOOL_CALL_FIELDS = ("tool_calls", "tools", "actions", "tool_traces", "steps")

# 路径里出现这些词更像对话端点
_PATH_HINTS = ("chat", "ask", "query", "agent", "run", "invoke", "diagnosis", "completion", "talk")
# 这些明显不是对话端点
_PATH_BLOCKLIST = ("login", "logout", "token", "ingest", "seed", "upload", "import", "webhook")

# 探测失败时的兜底配置（vehicle-diagnosis-agent）
DEFAULT_PROBE = {
    "chat_path": "/api/v1/diagnosis/chat",
    "request_field": "message",
    "reply_field": "reply",
    "tool_calls_field": "tool_calls",
}

_OPENAPI_CANDIDATES = ("/openapi.json", "/api/openapi.json", "/api/v1/openapi.json", "/v3/api-docs")


def normalize_base_url(base_url: str) -> str:
    """去掉尾斜杠与多余空白，保留 scheme + host[:port][/前缀]。"""
    url = (base_url or "").strip().rstrip("/")
    if url and not url.startswith(("http://", "https://")):
        url = "http://" + url
    return url


def _resolve_ref(schema: Any, spec: Dict[str, Any], depth: int = 0) -> Dict[str, Any]:
    """展开 ``{"$ref": "#/components/schemas/X"}``。"""
    if not isinstance(schema, dict) or depth > 5:
        return schema if isinstance(schema, dict) else {}
    ref = schema.get("$ref")
    if not isinstance(ref, str):
        return schema
    node: Any = spec
    for part in ref.lstrip("#/").split("/"):
        if not isinstance(node, dict):
            return {}
        node = node.get(part)
    return _resolve_ref(node, spec, depth + 1) if isinstance(node, dict) else {}


def _properties(schema: Any, spec: Dict[str, Any]) -> Dict[str, Any]:
    schema = _resolve_ref(schema, spec)
    props = schema.get("properties")
    return props if isinstance(props, dict) else {}


def _pick_field(props: Dict[str, Any], candidates: Tuple[str, ...],
                want_array: bool = False) -> Optional[str]:
    """按优先级挑字段；找不到就取第一个类型匹配的字段。"""
    for name in candidates:
        if name in props:
            return name

    for name, meta in props.items():
        if not isinstance(meta, dict):
            continue
        # anyOf / nullable 的字段类型藏在 anyOf 里
        types = [meta.get("type")]
        for branch in meta.get("anyOf") or []:
            if isinstance(branch, dict):
                types.append(branch.get("type"))
        types = [t for t in types if isinstance(t, str)]
        if want_array:
            if "array" in types:
                return name
        elif "string" in types:
            return name
    return None


def _iter_post_operations(spec: Dict[str, Any]):
    paths = spec.get("paths")
    if not isinstance(paths, dict):
        return
    for path, methods in paths.items():
        if not isinstance(methods, dict):
            continue
        for method, operation in methods.items():
            if str(method).lower() != "post" or not isinstance(operation, dict):
                continue
            yield path, operation


def _score_candidate(path: str, request_field: Optional[str], reply_field: Optional[str]) -> int:
    lowered = path.lower()
    if any(word in lowered for word in _PATH_BLOCKLIST):
        return -1
    score = 0
    if any(hint in lowered for hint in _PATH_HINTS):
        score += 10
    if request_field:
        score += 5
    if reply_field:
        score += 5
    # 同等条件下路径短的更可能是主入口
    score -= lowered.count("/")
    return score


def _extract_operation_fields(path: str, operation: Dict[str, Any],
                              spec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    body = (
        operation.get("requestBody", {})
        .get("content", {})
        .get("application/json", {})
        .get("schema", {})
    )
    request_props = _properties(body, spec)

    responses = operation.get("responses") or {}
    ok = responses.get("200") or responses.get("201") or responses.get("default") or {}
    response_props = _properties(
        ok.get("content", {}).get("application/json", {}).get("schema", {}), spec
    )
    # response_model 是 list[X] 的情况（没有 properties），跳过
    if not request_props and not response_props:
        return None

    request_field = _pick_field(request_props, PROMPT_FIELDS)
    reply_field = _pick_field(response_props, REPLY_FIELDS)
    tool_calls_field = _pick_field(response_props, TOOL_CALL_FIELDS, want_array=True)

    if not request_field or not reply_field:
        return None

    return {
        "chat_path": path,
        "request_field": request_field,
        "reply_field": reply_field,
        "tool_calls_field": tool_calls_field or "tool_calls",
        "_score": _score_candidate(path, request_field, reply_field),
    }


def _own_app_name() -> str:
    settings_name = ""
    try:
        from app.core.config import get_settings

        settings_name = str(get_settings().APP_NAME or "")
    except Exception:  # noqa: BLE001 - 配置不可用时退化为内置名
        pass
    return (settings_name or "agent-eval").strip().lower()


def detect_self_reference(base: str, service_title: str) -> Optional[str]:
    """判断目标地址是不是 agent-eval 自己。

    这是踩过的坑：agent-eval 默认占 8000，被测 Agent 也常占 8000，端口冲突时
    本机进程胜出，探测会一路绿灯地把自己识别成被测对象，跑出来的分数全是自己
    对自己的评价。这里做硬拦截，命中即 failure，而不是给个软提示。

    两种判定方式：

    1. 目标服务的 OpenAPI 标题与本服务 APP_NAME 相同；
    2. 目标端口与本服务实际监听端口相同（APP_PORT 由 ``app.main.__main__`` 写入环境变量）。
    """
    own_name = _own_app_name()
    title = (service_title or "").strip().lower()
    if own_name and title and title == own_name:
        return f"这个地址是 agent-eval 自己（目标服务标题为 “{service_title}”），评测自己没有意义"

    own_port = os.getenv("APP_PORT")
    if own_port and own_port.isdigit():
        try:
            target_port = urlparse(base).port
        except Exception:  # noqa: BLE001 - 地址格式异常时跳过端口比对
            target_port = None
        if target_port is not None and int(target_port) == int(own_port):
            return f"端口 {target_port} 正是 agent-eval 自己占用的端口，请填被测 Agent 的地址"
    return None


def probe_agent_service(base_url: str, timeout: float = 8.0) -> Dict[str, Any]:
    """探测一个 HTTP Agent 服务的对话端点。

    返回 ``{"ok": True, ...字段..., "candidates": [...]}``，
    失败时返回 ``{"ok": False, "error": "...", **DEFAULT_PROBE}``（仍带兜底配置，
    便于调用方"探测失败也按默认试一把"）。
    """
    base = normalize_base_url(base_url)
    if not base:
        return {"ok": False, "error": "地址为空", **DEFAULT_PROBE}

    try:
        import httpx
    except ImportError:  # pragma: no cover
        return {"ok": False, "error": "缺少 httpx", **DEFAULT_PROBE}

    spec: Optional[Dict[str, Any]] = None
    last_error = ""
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        for doc_path in _OPENAPI_CANDIDATES:
            try:
                resp = client.get(base + doc_path)
                if resp.status_code == 200:
                    payload = resp.json()
                    if isinstance(payload, dict) and payload.get("paths"):
                        spec = payload
                        break
            except Exception as exc:  # noqa: BLE001 - 逐个候选尝试，失败继续
                last_error = str(exc)

    if spec is None:
        message = last_error or f"未找到 OpenAPI 文档（已尝试 {', '.join(_OPENAPI_CANDIDATES)}）"
        logger.info("探测 %s 失败: %s", base, message)
        return {"ok": False, "error": message, **DEFAULT_PROBE}

    service_title = (spec.get("info") or {}).get("title") or ""

    # 自指拦截优先于任何端点识别：评自己是无效评测，必须让用户明确看到
    self_reference = detect_self_reference(base, service_title)
    if self_reference:
        logger.info("探测 %s 命中自指，已拦截: %s", base, self_reference)
        return {
            **DEFAULT_PROBE,
            "ok": False,
            "base_url": base,
            "error": self_reference,
            "self_reference": True,
            "service_title": service_title,
        }

    candidates: List[Dict[str, Any]] = []
    for path, operation in _iter_post_operations(spec):
        fields = _extract_operation_fields(path, operation, spec)
        if fields:
            candidates.append(fields)

    if not candidates:
        return {"ok": False, "error": "OpenAPI 里没有可用的 POST 对话端点", **DEFAULT_PROBE}

    candidates.sort(key=lambda item: item["_score"], reverse=True)
    best = {k: v for k, v in candidates[0].items() if not k.startswith("_")}
    result: Dict[str, Any] = {"ok": True, "base_url": base, **best}
    result["candidates"] = [
        {k: v for k, v in c.items() if not k.startswith("_")} for c in candidates[:5]
    ]
    result["service_title"] = service_title
    logger.info("探测 %s -> %s", base, best["chat_path"])
    return result
