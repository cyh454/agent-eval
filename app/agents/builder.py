"""被测 Agent 的构建器。

对外只暴露一个类方法语义：``AgentBuilder().build(config) -> agent``，
返回的 agent 具备 ``invoke(payload, config=...)`` 接口，与 ``TrackedAgentExecutor`` 对接。

支持三种用法：
1. ``config`` 里给纯字典（model / temperature / system_prompt / tools），内部自动构造 ChatOpenAI；
2. ``config["llm"]`` 直接传入已构造好的 LLM 实例（便于自定义或单元测试注入假模型）；
3. ``config["type"] = "remote_http"`` 时改为包装一个已有的 Agent HTTP 服务，
   不构造本地 Agent，详见 ``app.agents.adapters``。
"""

import ast
import logging
import operator as op
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence
from uuid import uuid4

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool, StructuredTool

from app.agents.adapters import build_remote_agent
from app.core.config import get_settings
from app.core.llm import build_chat_model

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "你是一个乐于助人的 AI 助手。请准确、完整地回答用户问题，"
    "需要时可以调用提供的工具来获取信息。"
)


# --------------------------------------------------------------------------
# 内置安全工具（只读、无副作用，便于任务完成度评估中的「工具使用正确性」维度）
# --------------------------------------------------------------------------

_BIN_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
}
_UNARY_OPS = {ast.UAdd: op.pos, ast.USub: op.neg}


def _safe_calculate(expression: str) -> str:
    """只允许数字与基础运算符的表达式求值，避免任意代码执行。"""
    try:
        tree = ast.parse(str(expression), mode="eval")
    except SyntaxError as exc:
        return f"表达式无法解析: {exc.msg}"

    def _visit(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            left, right = _visit(node.left), _visit(node.right)
            if isinstance(node.op, ast.Pow) and (abs(right) > 8 or abs(left) > 1e6):
                raise ValueError("幂运算超出安全范围")
            return _BIN_OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](_visit(node.operand))
        raise ValueError("表达式包含不允许的语法")

    try:
        return str(_visit(tree))
    except ZeroDivisionError:
        return "计算错误: 除数为零"
    except Exception as exc:  # noqa: BLE001 - 工具返回值需对用户友好
        return f"计算错误: {exc}"


def _current_time(_: str = "") -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _text_length(text: str) -> str:
    return f"字符数: {len(text)}, 单词数: {len(text.split())}"


TOOL_REGISTRY: Dict[str, Any] = {
    "calculator": lambda: StructuredTool.from_function(
        func=_safe_calculate,
        name="calculator",
        description="计算数学表达式，例如 '(12 + 8) * 3'。输入是一个算术表达式字符串。",
    ),
    "current_time": lambda: StructuredTool.from_function(
        func=_current_time,
        name="current_time",
        description="获取当前日期和时间。",
    ),
    "text_length": lambda: StructuredTool.from_function(
        func=_text_length,
        name="text_length",
        description="统计给定文本的字符数与单词数。",
    ),
}


def available_tool_names() -> List[str]:
    return sorted(TOOL_REGISTRY)


class SimpleAgent:
    """最小可用的工具调用型 Agent。

    无工具时直接单次调用 LLM；有工具时执行最多 ``max_iterations`` 轮工具循环，
    并把工具的开始 / 结束 / 错误事件回传给传入的 callback handler，
    这样 ``AgentExecutionTracker`` 能完整记录工具调用轨迹。
    """

    def __init__(
        self,
        llm: Any,
        *,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        tools: Optional[Sequence[BaseTool]] = None,
        max_iterations: int = 6,
    ) -> None:
        self.llm = llm
        self.system_prompt = system_prompt
        self.tools: List[BaseTool] = list(tools or [])
        self.max_iterations = max(1, int(max_iterations))
        self.tools_by_name: Dict[str, BaseTool] = {t.name: t for t in self.tools}
        self._bound_llm = self.llm.bind_tools(self.tools) if self.tools else self.llm

    @property
    def model_name(self) -> Optional[str]:
        return getattr(self.llm, "model_name", None) or getattr(self.llm, "model", None)

    @staticmethod
    def _extract_input(payload: Any) -> str:
        if isinstance(payload, dict):
            for key in ("input", "input_prompt", "question", "prompt"):
                if key in payload:
                    return str(payload[key])
            return str(payload)
        if isinstance(payload, str):
            return payload
        return str(payload)

    @staticmethod
    def _handlers(config: Optional[Dict[str, Any]]) -> List[Any]:
        if not config:
            return []
        handlers = config.get("callbacks") or []
        return list(handlers) if isinstance(handlers, (list, tuple)) else [handlers]

    def _notify(self, handler: Any, method: str, *args: Any) -> None:
        func = getattr(handler, method, None)
        if func is None:
            return
        try:
            func(*args, run_id=uuid4())
        except Exception as exc:  # noqa: BLE001 - 回调异常不应中断 Agent 执行
            logger.debug("回调 %s 执行失败: %s", method, exc)

    def _run_tool(self, tool_call: Dict[str, Any], messages: List[BaseMessage], handlers: List[Any]) -> None:
        name = tool_call.get("name", "")
        args = tool_call.get("args", {}) or {}
        normalized_args = args if isinstance(args, dict) else {"input": args}
        serialized = {"name": name, "description": ""}
        input_repr = str(normalized_args)

        tool = self.tools_by_name.get(name)
        for handler in handlers:
            self._notify(handler, "on_tool_start", serialized, input_repr)

        if tool is None:
            output = f"未找到工具: {name}"
            for handler in handlers:
                self._notify(handler, "on_tool_error", ValueError(output))
        else:
            try:
                output = str(tool.invoke(normalized_args))
                for handler in handlers:
                    self._notify(handler, "on_tool_end", output)
            except Exception as exc:  # noqa: BLE001 - 工具异常要转成 ToolMessage 传给模型
                output = f"工具执行失败: {exc}"
                for handler in handlers:
                    self._notify(handler, "on_tool_error", exc)

        messages.append(
            ToolMessage(
                content=output,
                tool_call_id=tool_call.get("id") or str(uuid4()),
                name=name,
            )
        )

    def invoke(self, payload: Any, config: Optional[Dict[str, Any]] = None) -> str:
        """执行一次 Agent 调用，返回最终文本。"""
        user_input = self._extract_input(payload)
        handlers = self._handlers(config)
        messages: List[BaseMessage] = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_input),
        ]

        if not self.tools:
            response = self.llm.invoke(messages, config=config)
            return getattr(response, "content", None) or str(response)

        for _ in range(self.max_iterations):
            response = self._bound_llm.invoke(messages, config=config)
            if not isinstance(response, AIMessage):
                return getattr(response, "content", None) or str(response)

            messages.append(response)
            tool_calls: Iterable[Dict[str, Any]] = getattr(response, "tool_calls", None) or []
            tool_calls = list(tool_calls)

            if not tool_calls:
                return response.content or ""

            for tool_call in tool_calls:
                self._run_tool(tool_call, messages, handlers)

        # 达到轮次上限，禁用工具做一次收尾，保证有最终答复
        final = self.llm.invoke(messages, config=config)
        return getattr(final, "content", None) or str(final)


class AgentBuilder:
    """根据配置构建被测 Agent。"""

    def __init__(self, default_config: Optional[Dict[str, Any]] = None) -> None:
        self.default_config = default_config or {}

    def _resolve_tools(self, raw_tools: Any) -> List[BaseTool]:
        tools: List[BaseTool] = []
        unknown: List[str] = []

        for item in raw_tools or []:
            if isinstance(item, BaseTool):
                tools.append(item)
                continue
            if isinstance(item, dict):
                tools.append(StructuredTool.from_function(**item))
                continue
            factory = TOOL_REGISTRY.get(str(item))
            if factory is not None:
                tools.append(factory())
            else:
                unknown.append(str(item))

        if unknown:
            logger.warning(
                "未知的 Agent 工具 %s，已忽略；可用内置工具: %s",
                unknown,
                available_tool_names(),
            )
        return tools

    # 走 HTTP 调用已有 Agent 服务时的类型标识
    REMOTE_TYPES = {"remote_http", "http", "remote"}

    def build(self, agent_config: Optional[Dict[str, Any]] = None) -> Any:
        merged: Dict[str, Any] = {**self.default_config, **(agent_config or {})}

        # 外部 Agent 服务：不构造本地 SimpleAgent，改为包装远端 HTTP 接口
        agent_type = str(merged.get("type") or merged.get("agent_type") or "").lower()
        if agent_type in self.REMOTE_TYPES:
            return build_remote_agent(merged, default_model=get_settings().LLM_MODEL)

        llm = merged.get("llm") or build_chat_model(merged)
        tools = self._resolve_tools(merged.get("tools") or merged.get("tool_names"))

        return SimpleAgent(
            llm,
            system_prompt=merged.get("system_prompt") or DEFAULT_SYSTEM_PROMPT,
            tools=tools,
            max_iterations=int(merged.get("max_iterations", 6)),
        )
