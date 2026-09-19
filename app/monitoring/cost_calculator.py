"""成本计算器 —— 全局唯一的模型价格来源。

原先的问题：``performance.py`` 自带一份 ``PRICING``（键为 prompt/completion），
``cost_calculator.py`` 另有一份（键为 input/output），两份价格互不一致且都已经过时。
这里统一为一份表，键名统一为 ``prompt`` / ``completion``。
"""

from functools import lru_cache
from typing import Any, Dict, Mapping, Optional

# 统一使用「美元 / 1K tokens」为单位。
# OpenAI 价格参考 openai.com/pricing；智谱价格为官方人民币价按汇率折算，仅供估算。
PRICING: Dict[str, Dict[str, float]] = {
    # OpenAI
    "gpt-4": {"prompt": 0.03, "completion": 0.06},
    "gpt-4-turbo": {"prompt": 0.01, "completion": 0.03},
    "gpt-4-turbo-preview": {"prompt": 0.01, "completion": 0.03},
    "gpt-4o": {"prompt": 0.005, "completion": 0.015},
    "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.0006},
    "gpt-3.5-turbo": {"prompt": 0.0005, "completion": 0.0015},
    "gpt-3.5-turbo-16k": {"prompt": 0.003, "completion": 0.004},
    # 智谱 GLM（OpenAI 兼容端点）
    # 官方报价单位是「元 / 百万 tokens」，本表单位是「美元 / 1K tokens」：
    #   USD_per_1K = (CNY_per_M / 1000) / USD_CNY
    # 换算汇率取 1 USD ≈ 7.15 CNY。输出按官方短输出档（[0, 0.2K)）计价，
    # 长输出档价格更高，详见 https://docs.bigmodel.cn/cn/guide/start/pricing
    "glm-4.5-air": {"prompt": 0.000112, "completion": 0.000280},
    "glm-4-plus": {"prompt": 0.000699, "completion": 0.000699},
    "glm-4-air-250414": {"prompt": 0.000070, "completion": 0.000070},
    "glm-4-flash": {"prompt": 0.0, "completion": 0.0},
    # Embedding
    "text-embedding-ada-002": {"prompt": 0.0001, "completion": 0.0},
    "text-embedding-3-small": {"prompt": 0.00002, "completion": 0.0},
    "text-embedding-3-large": {"prompt": 0.00013, "completion": 0.0},
    "embedding-3": {"prompt": 0.000070, "completion": 0.0},
    "embedding-2": {"prompt": 0.000070, "completion": 0.0},
}

# 成本计算的兜底模型：agent_config 未指定 model 时，按它的单价计价。
# 必须与 Settings.LLM_MODEL 的默认值保持一致——用一个你根本没在跑的模型计价，
# 算出来的成本就是错的（比如 glm-4-plus 比 glm-4.5-air 贵 2.5~6 倍）。
DEFAULT_MODEL = "glm-4.5-air"
_DEFAULT_PRICING = PRICING[DEFAULT_MODEL]


def normalize_pricing(raw: Mapping[str, Mapping[str, float]]) -> Dict[str, Dict[str, float]]:
    """兼容历史键名 input/output，归一化为 prompt/completion。"""
    normalized: Dict[str, Dict[str, float]] = {}
    for model, prices in raw.items():
        normalized[model] = {
            "prompt": prices.get("prompt", prices.get("input", 0.0)),
            "completion": prices.get("completion", prices.get("output", 0.0)),
        }
    return normalized


class CostCalculator:
    """模型调用成本估算。"""

    def __init__(self, custom_pricing: Optional[Mapping[str, Mapping[str, float]]] = None) -> None:
        base = normalize_pricing(PRICING)
        if custom_pricing:
            base.update(normalize_pricing(custom_pricing))
        self.pricing = base

    def get_pricing(self, model: Optional[str]) -> Dict[str, float]:
        if model and model in self.pricing:
            return self.pricing[model]
        return self.pricing.get(DEFAULT_MODEL, _DEFAULT_PRICING)

    def calculate_cost(self, model: Optional[str], prompt_tokens: int, completion_tokens: int) -> float:
        """按 1K tokens 计费，返回美元成本。"""
        pricing = self.get_pricing(model)
        prompt_cost = (max(prompt_tokens, 0) / 1000) * pricing["prompt"]
        completion_cost = (max(completion_tokens, 0) / 1000) * pricing["completion"]
        return prompt_cost + completion_cost

    def calculate_embedding_cost(self, model: Optional[str], tokens: int) -> float:
        pricing = self.get_pricing(model)
        return (max(tokens, 0) / 1000) * pricing["prompt"]

    def estimate_tokens(self, text: str, model: str = DEFAULT_MODEL) -> int:
        """估算 Token 数：优先 tiktoken，失败时回退到按字符比例的粗估。"""
        if not text:
            return 0

        encoding = self._get_encoding(model)
        if encoding is not None:
            try:
                return len(encoding.encode(text))
            except Exception:  # noqa: BLE001 - 编码异常时回退到估算
                pass

        chinese_chars = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
        other_chars = len(text) - chinese_chars
        return int(chinese_chars / 2 + other_chars / 4)

    @staticmethod
    @lru_cache(maxsize=8)
    def _get_encoding(model: str) -> Any:
        try:
            import tiktoken
        except ImportError:
            return None
        try:
            return tiktoken.encoding_for_model(model)
        except Exception:  # noqa: BLE001 - 未知模型统一使用通用编码
            try:
                return tiktoken.get_encoding("cl100k_base")
            except Exception:  # noqa: BLE001
                return None

    def format_cost(self, cost: float) -> str:
        if cost >= 1:
            return f"${cost:.2f}"
        if cost >= 0.01:
            return f"${cost:.4f}"
        return f"${cost:.6f}"

    def supported_models(self) -> Dict[str, Dict[str, float]]:
        return dict(self.pricing)
