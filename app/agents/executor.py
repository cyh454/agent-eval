"""带链路追踪的 Agent 执行器。

追踪器负责补齐三类信息，缺一样性能维度就会失真：

1. 端到端耗时：只在首跳记录起点，否则多轮执行算出来的只剩最后一跳；
2. Token 用量：优先取服务端上报，拿不到才按可见文本估算；
3. 用量来源标记 ``token_source``（``remote`` / ``estimated``），
   供性能维度决定这些数字能不能当真，详见 ``app.monitoring.token_estimator``。
"""

import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import BaseCallbackHandler

from app.monitoring.cost_calculator import CostCalculator


def _extract_token_usage(response: Any) -> Dict[str, int]:
    """从不同版本的 LangChain 响应中提取 Token 用量。"""
    usage: Dict[str, Any] = {}

    metadata = getattr(response, "response_metadata", None) or {}
    if isinstance(metadata, dict):
        usage = (
            metadata.get("token_usage")
            or metadata.get("usage")
            or metadata.get("model_stats")
            or {}
        )

    # LangChain 新版统一在 usage_metadata: {"input_tokens":..,"output_tokens":..}
    if not usage:
        usage_metadata = getattr(response, "usage_metadata", None)
        if isinstance(usage_metadata, dict) and usage_metadata:
            usage = {
                "prompt_tokens": usage_metadata.get("input_tokens", 0),
                "completion_tokens": usage_metadata.get("output_tokens", 0),
            }

    llm_output = getattr(response, "llm_output", None)
    if not usage and isinstance(llm_output, dict):
        usage = llm_output.get("token_usage", {}) or {}

    prompt = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
    completion = usage.get("completion_tokens") or usage.get("output_tokens") or 0
    total = usage.get("total_tokens")

    return {
        "prompt_tokens": int(prompt),
        "completion_tokens": int(completion),
        "total_tokens": int(total) if total is not None else int(prompt) + int(completion),
    }


class AgentExecutionTracker(BaseCallbackHandler):
    """Agent 执行追踪器 - 记录完整执行过程。"""

    def __init__(self) -> None:
        self.steps: List[Dict[str, Any]] = []
        self.tool_calls: List[Dict[str, Any]] = []
        self.intermediate_outputs: List[Any] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.token_source: Optional[str] = None
        self.latency_source: str = "client"

    # ---------------- LLM ----------------
    def on_llm_start(self, serialized: Dict[str, Any], prompts: Any, **kwargs: Any) -> None:
        # 只在首跳记录起点，覆盖整个多轮执行过程
        if self.start_time is None:
            self.start_time = time.time()

        self.steps.append(
            {
                "type": "llm_start",
                "timestamp": datetime.utcnow().isoformat(),
                "prompts": list(prompts) if isinstance(prompts, (list, tuple)) else prompts,
            }
        )

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        duration = (time.time() - self.start_time) if self.start_time else 0.0

        metadata = getattr(response, "response_metadata", None) or {}
        if not isinstance(metadata, dict):
            metadata = {}

        # 服务端自测耗时优先：客户端的秒表混着网络往返和请求排队
        if self._read_server_latency(metadata) is not None:
            duration = self._read_server_latency(metadata) / 1000.0
            self.latency_source = "server"
        else:
            self.latency_source = "client"

        source = metadata.get("token_source")
        if source:
            self.token_source = str(source)

        token_usage = _extract_token_usage(response)
        self.prompt_tokens += token_usage["prompt_tokens"]
        self.completion_tokens += token_usage["completion_tokens"]
        self.total_tokens += token_usage["total_tokens"]

        content = getattr(response, "content", None)
        if content is None:
            generations = getattr(response, "generations", None) or []
            if generations:
                first = generations[0][0] if isinstance(generations[0], list) else generations[0]
                content = getattr(first, "text", None) or getattr(first, "message", None)
        if content is None:
            content = str(response)

        self.steps.append(
            {
                "type": "llm_end",
                "timestamp": datetime.utcnow().isoformat(),
                "duration": duration,
                "response": content,
                "token_usage": token_usage,
            }
        )
        # 用服务端耗时刻画时间轴，保证 get_execution_summary 的 total_duration 与之一致
        self.end_time = (
            (self.start_time or time.time()) + duration
            if self.latency_source == "server"
            else time.time()
        )

    @staticmethod
    def _read_server_latency(metadata: Dict[str, Any]) -> Optional[float]:
        latency = metadata.get("latency_ms")
        if isinstance(latency, (int, float)) and latency > 0:
            return float(latency)
        return None

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        self.end_time = time.time()
        self.steps.append(
            {
                "type": "llm_error",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(error),
            }
        )

    # ---------------- Tool ----------------
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> None:
        self.tool_calls.append(
            {
                "name": serialized.get("name", "unknown") if isinstance(serialized, dict) else "unknown",
                "arguments": input_str,
                "start_time": datetime.utcnow().isoformat(),
                "status": "running",
            }
        )

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        if self.tool_calls:
            self.tool_calls[-1].update(
                {
                    "status": "completed",
                    "end_time": datetime.utcnow().isoformat(),
                    "output": output if isinstance(output, str) else str(output),
                }
            )

    def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
        if self.tool_calls:
            self.tool_calls[-1].update(
                {
                    "status": "failed",
                    "end_time": datetime.utcnow().isoformat(),
                    "error": str(error),
                }
            )

    # ---------------- 摘要 ----------------
    def get_execution_summary(self) -> Dict[str, Any]:
        if self.start_time and self.end_time:
            total_duration = self.end_time - self.start_time
        elif self.start_time:
            total_duration = time.time() - self.start_time
        else:
            total_duration = 0.0

        return {
            "steps": self.steps,
            "tool_calls": self.tool_calls,
            "intermediate_outputs": self.intermediate_outputs,
            "total_duration": total_duration,
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "token_source": self.token_source,
            "latency_source": self.latency_source,
        }


class TrackedAgentExecutor:
    """带追踪的 Agent 执行器。"""

    def __init__(self, agent_chain: Any, model_name: str = "unknown") -> None:
        self.agent_chain = agent_chain
        self.model_name = model_name
        self.cost_calculator = CostCalculator()

    def execute(self, input_prompt: str) -> Dict[str, Any]:
        """执行 Agent 并采集执行轨迹、Token 用量与估算成本。"""
        tracker = AgentExecutionTracker()

        try:
            # 兼容 dict / str 两种调用约定
            if self._accepts_dict():
                payload: Any = {"input": input_prompt}
            else:
                payload = input_prompt

            result = self.agent_chain.invoke(payload, config={"callbacks": [tracker]})
        except TypeError:
            # 少数 chain 不接受 config 关键字，降级为无回调调用
            tracker = AgentExecutionTracker()
            try:
                result = self.agent_chain.invoke(input_prompt)
            except Exception as exc:  # noqa: BLE001
                return self._failure(tracker, exc)
        except Exception as exc:  # noqa: BLE001 - 单个用例失败不应中断整批评估
            return self._failure(tracker, exc)

        summary = tracker.get_execution_summary()
        final_response = self._stringify(result)

        return {
            "status": "completed",
            "final_response": final_response,
            "execution_summary": summary,
            "estimated_cost": self.cost_calculator.calculate_cost(
                self.model_name, summary["prompt_tokens"], summary["completion_tokens"]
            ),
        }

    # ---------------- helpers ----------------
    def _accepts_dict(self) -> bool:
        """判断被测 Agent 是否接受 dict 入参。"""
        invoke = getattr(self.agent_chain, "invoke", None)
        if invoke is None:
            return False
        annotations = getattr(invoke, "__annotations__", {}) or {}
        first_param = next(iter(annotations), None)
        return first_param in {"payload", "input", "input_dict"}

    @staticmethod
    def _stringify(result: Any) -> str:
        if isinstance(result, str):
            return result
        content = getattr(result, "content", None)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict) and block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
            if parts:
                return "".join(parts)
        if isinstance(result, dict):
            for key in ("output", "result", "answer", "text"):
                if key in result:
                    return str(result[key])
        return str(result)

    def _failure(self, tracker: AgentExecutionTracker, exc: BaseException) -> Dict[str, Any]:
        return {
            "status": "failed",
            "error": str(exc),
            "execution_summary": tracker.get_execution_summary(),
            "estimated_cost": 0.0,
        }
