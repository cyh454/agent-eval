"""性能评估器：延迟、成本与 Token 效率。

价格统一取自 CostCalculator；模型名优先取执行记录里的
``execution_metadata.model_name``，取不到才回退到配置值——
被测 Agent 的显示名不能当模型名用，拿它查价格必然查不到。
"""

import logging
from typing import Any, Dict, Optional

from app.core.config import get_settings
from app.evaluators.base import BaseEvaluator
from app.models.datasets import TestCase
from app.models.execution import AgentExecution
from app.monitoring.cost_calculator import CostCalculator

logger = logging.getLogger(__name__)


class PerformanceEvaluator(BaseEvaluator):
    """性能评估器"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.cost_calculator = CostCalculator(config.get("pricing") if config else None)
        self.default_model = (config or {}).get("model") or get_settings().LLM_MODEL

    @staticmethod
    def _resolve_model(execution: AgentExecution, fallback: str) -> str:
        metadata = getattr(execution, "execution_metadata", None) or {}
        if isinstance(metadata, dict) and metadata.get("model_name"):
            return str(metadata["model_name"])
        return fallback

    def evaluate(self, execution: AgentExecution, test_case: TestCase) -> Dict[str, Any]:
        settings = get_settings()

        prompt_tokens = execution.prompt_tokens or 0
        completion_tokens = execution.completion_tokens or 0
        total_tokens = execution.total_tokens or 0

        # 计算延迟
        latency = execution.total_duration or 0.0

        # 计算成本（统一价格表）
        model_name = self._resolve_model(execution, self.default_model)
        total_cost = self.cost_calculator.calculate_cost(
            model_name, prompt_tokens, completion_tokens
        )

        # 计算 Token 效率（响应长度 / 总Token数）
        response_text = execution.final_response or ""
        token_efficiency = (len(response_text) / total_tokens) if total_tokens > 0 else 0.0

        # 根据预设阈值评分
        latency_threshold = float(self.config.get("latency_threshold", settings.DEFAULT_LATENCY_THRESHOLD))
        cost_threshold = float(self.config.get("cost_threshold", settings.DEFAULT_COST_THRESHOLD))

        if latency_threshold <= 0 or cost_threshold <= 0:
            logger.warning("性能阈值必须为正数，已回退到默认配置")
            latency_threshold = settings.DEFAULT_LATENCY_THRESHOLD
            cost_threshold = settings.DEFAULT_COST_THRESHOLD

        latency_score = max(0.0, 1 - (latency / latency_threshold))
        cost_score = max(0.0, 1 - (total_cost / cost_threshold))

        metadata = getattr(execution, "execution_metadata", None) or {}
        token_source = metadata.get("token_source")
        latency_source = metadata.get("latency_source") or "client"

        # 没有可信的 token 用量时，三项加权会失真：cost 恒 0 白送 30% 权重，
        # token_efficiency 恒 0 白丢 20% 权重，最后只剩门槛而被当成完整测评。
        # 这种情况把权重全部让给确实测到的延迟，并在 metrics / reasoning 里写明降级。
        has_tokens = bool(token_source) and total_tokens > 0

        if has_tokens:
            performance_score = (
                latency_score * 0.5 + cost_score * 0.3 + min(1.0, token_efficiency) * 0.2
            )
        else:
            performance_score = latency_score

        notes: list = []
        if latency_source == "server":
            notes.append("延迟取被测服务端自测值")
        else:
            notes.append("延迟取客户端计时（含网络往返）")
        if token_source == "estimated":
            notes.append("Token 为本地估算，不含被测 Agent 内部上下文，仅供横向比较")
        elif token_source == "remote":
            notes.append("Token 由被测服务端上报")
        else:
            notes.append("未获得 Token 用量，性能维度仅按延迟计分")

        return {
            "score": max(0.0, min(1.0, performance_score)),
            "metrics": {
                "latency": latency,
                "throughput": 0.0,  # 需在整体评估结束后由 MetricsCollector 计算
                "token_efficiency": token_efficiency if has_tokens else None,
                "cost": total_cost if has_tokens else None,
                "model_name": model_name,
                "total_tokens": total_tokens,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "token_source": token_source,
                "latency_source": latency_source,
                "performance_incomplete": not has_tokens,
            },
            "reasoning": (
                f"Model: {model_name}, Latency: {latency:.2f}s, "
                f"Cost: ${total_cost:.4f}, Token Efficiency: {token_efficiency:.2f}。"
                + "；".join(notes)
                + "。"
            ),
        }

    def get_dimension_name(self) -> str:
        return "performance"
