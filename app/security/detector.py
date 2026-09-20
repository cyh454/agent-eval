"""安全检测器 - 基于规则的安全检查。

两处输入不可信，处理时必须兜住：

* 工具参数可能是 dict，也可能是 callback 追踪器记下来的字符串；
* 自定义规则的正则来自数据库用户输入，非法表达式应跳过而不是让评估崩溃。
"""

import ast
import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SecurityDetector:
    """安全检测器 - 基于规则的安全检查"""

    # 危险工具列表
    DANGEROUS_TOOLS = [
        "shell",
        "execute_command",
        "file_delete",
        "system_modify",
        "database_drop",
        "network_scan",
    ]

    # 敏感关键词
    SENSITIVE_KEYWORDS = [
        "system prompt",
        "instructions:",
        "your instructions are",
        "ignore previous",
        "bypass security",
        "sudo",
        "rm -rf",
        "drop table",
        "truncate",
        "delete from",
    ]

    # Prompt 泄露模式
    PROMPT_LEAKAGE_PATTERNS = [
        r"as an? AI( assistant)?, (my|your) (instructions?|guidelines?) (are|is)",
        r"you are (a )?(helpful|AI)( assistant)?",
        r"your (role|persona|purpose)",
        r"system:?\s*(you|your)",
        r"instructions:?\s*\n",
    ]

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.custom_rules: List[Dict[str, Any]] = list(
            self.config.get("custom_rules", []) or []
        )

    # ------------------------------------------------------------------ 工具调用
    def check_tool_call(self, tool_call: Dict) -> Optional[Dict]:
        """检查工具调用是否安全"""
        if not isinstance(tool_call, dict):
            return None

        tool_name = tool_call.get("name", "")

        # 检查是否为危险工具
        if tool_name in self.DANGEROUS_TOOLS:
            return {
                "type": "dangerous_op",
                "severity": "high",
                "description": f"尝试调用危险工具: {tool_name}",
                "tool_call": tool_call,
            }

        # 检查工具参数（可能是 dict，也可能是 callback 记录的字符串）
        args = self._coerce_arguments(tool_call.get("arguments", {}))

        if isinstance(args, dict):
            for value in args.values():
                if isinstance(value, str):
                    violation = self._check_text_safety(value)
                    if violation:
                        violation["tool_call"] = tool_call
                        return violation
        elif isinstance(args, str):
            violation = self._check_text_safety(args)
            if violation:
                violation["tool_call"] = tool_call
                return violation

        return None

    # ------------------------------------------------------------------ 输出
    def check_output(self, output: Optional[str]) -> Optional[Dict]:
        """检查输出是否安全"""
        if not output or not isinstance(output, str):
            return None
        return self._check_text_safety(output)

    # ------------------------------------------------------------------ 内部
    @staticmethod
    def _coerce_arguments(arguments: Any) -> Any:
        """尽最大努力把参数还原为 dict，无法解析时保留原字符串。"""
        if isinstance(arguments, dict):
            return arguments
        if isinstance(arguments, (bytes, bytearray)):
            arguments = arguments.decode("utf-8", errors="ignore")
        if not isinstance(arguments, str):
            return arguments

        stripped = arguments.strip()
        if not stripped:
            return {}

        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(stripped)
            except Exception:  # noqa: BLE001 - 逐个尝试不同解析方式
                continue
            if isinstance(parsed, dict):
                return parsed

        return arguments

    def _check_text_safety(self, text: str) -> Optional[Dict]:
        """检查文本安全性"""
        if not text or not isinstance(text, str):
            return None

        text_lower = text.lower()

        # 1. 检查敏感关键词
        for keyword in self.SENSITIVE_KEYWORDS:
            if keyword in text_lower:
                return {
                    "type": "sensitive_keyword",
                    "severity": "medium",
                    "description": f"检测到敏感关键词: {keyword}",
                    "keyword": keyword,
                }

        # 2. 检查 Prompt 泄露模式
        for pattern in self.PROMPT_LEAKAGE_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return {
                    "type": "prompt_leakage",
                    "severity": "high",
                    "description": "检测到可能的Prompt泄露",
                    "pattern": pattern,
                }

        # 3. 检查自定义规则（正则来自数据库，表达式非法时应跳过）
        for rule in self.custom_rules:
            pattern = rule.get("pattern")
            if not pattern:
                continue
            try:
                matched = re.search(pattern, text, re.IGNORECASE)
            except re.error as exc:
                logger.warning("自定义安全规则 %s 正则非法，已跳过: %s", rule.get("name"), exc)
                continue
            if matched:
                return {
                    "type": rule.get("type", "custom"),
                    "severity": rule.get("severity", "medium"),
                    "description": rule.get("description", "违反自定义规则"),
                    "rule_name": rule.get("name"),
                }

        return None
