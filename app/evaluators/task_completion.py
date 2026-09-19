import json
import logging
from typing import Any, Dict, List

from app.core.llm import (
    build_chat_model,
    parse_bool_response,
    parse_float_response,
)
from app.evaluators.base import BaseEvaluator
from app.models.datasets import TestCase
from app.models.execution import AgentExecution

logger = logging.getLogger(__name__)


class TaskCompletionEvaluator(BaseEvaluator):
    """任务完成度评估器 - 特别适用于多步骤Agent"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.llm = build_chat_model(config)
    
    def evaluate(self, execution: AgentExecution, test_case: TestCase) -> Dict[str, Any]:
        """评估任务完成度"""
        
        # 1. 检查工具使用
        tool_correctness = self._evaluate_tool_usage(execution, test_case)
        
        # 2. 检查步骤正确性
        steps_correctness = self._evaluate_steps(execution, test_case)
        
        # 3. 检查最终结果
        task_success = self._evaluate_final_result(execution, test_case)
        
        # 4. 计算综合得分
        task_score = (
            tool_correctness * 0.4 +
            steps_correctness * 0.3 +
            (1.0 if task_success else 0.0) * 0.3
        )
        
        return {
            "score": task_score,
            "metrics": {
                "task_success": task_success,
                "steps_correctness": steps_correctness,
                "tool_usage_correctness": tool_correctness,
                "expected_tools": test_case.expected_tools,
                "actual_tools": [tc.get("name") for tc in execution.tool_calls],
                "expected_steps": test_case.expected_steps,
                "actual_steps": len(execution.steps)
            },
            "reasoning": self._generate_reasoning(
                tool_correctness, steps_correctness, task_success
            )
        }
    
    def _evaluate_tool_usage(self, execution: AgentExecution, 
                           test_case: TestCase) -> float:
        """评估工具使用正确性"""
        
        expected_tools = set(test_case.expected_tools or [])
        actual_tools = {tc.get("name") for tc in execution.tool_calls}
        
        if not expected_tools:
            return 1.0  # 如果不期望任何工具，给满分
        
        # 计算交集比例
        correct_tools = expected_tools & actual_tools
        if len(expected_tools) == 0:
            return 1.0
        
        precision = len(correct_tools) / len(actual_tools) if actual_tools else 0
        recall = len(correct_tools) / len(expected_tools)
        
        # F1 score
        if precision + recall == 0:
            return 0.0
        
        return 2 * precision * recall / (precision + recall)
    
    def _evaluate_steps(self, execution: AgentExecution, 
                       test_case: TestCase) -> float:
        """评估步骤正确性"""
        
        expected_steps = test_case.expected_steps or []
        actual_steps = execution.steps or []
        
        if not expected_steps:
            return 1.0
        
        # 使用 LLM 评估步骤相似性
        if len(actual_steps) > 0:
            prompt = f"""请评估以下Agent的执行步骤是否符合预期：

【期望步骤】
{json.dumps(expected_steps, ensure_ascii=False, indent=2)}

【实际步骤】
{json.dumps(actual_steps, ensure_ascii=False, indent=2)}

请给出一个0-1的分数，表示步骤正确性。
只返回数字，不要其他内容。"""
            
            try:
                result = self.llm.invoke(prompt)
                score = parse_float_response(getattr(result, "content", None), default=0.0)
                if score > 0:
                    return score
            except Exception:  # noqa: BLE001 - LLM 失败时降级为数量比对
                logger.warning("步骤正确性 LLM 评估失败，回退到数量比对", exc_info=True)
        
        # 简单比较：步骤数量和关键步骤是否匹配
        expected_count = len(expected_steps)
        actual_count = len(actual_steps)
        
        count_score = 1.0 - abs(expected_count - actual_count) / max(expected_count, 1)
        
        return max(0.0, count_score)
    
    def _evaluate_final_result(self, execution: AgentExecution, 
                             test_case: TestCase) -> bool:
        """评估最终结果是否正确"""
        
        # 使用 LLM 评估最终结果
        prompt = f"""请评估以下Agent的最终回答是否正确解决了用户问题：

【用户问题】
{test_case.input_prompt}

【Agent最终回答】
{execution.final_response}

【参考答案】
{test_case.ground_truth}

请回答"true"或"false"，表示Agent是否成功完成了任务。"""
        
        try:
            result = self.llm.invoke(prompt)
            return parse_bool_response(getattr(result, "content", None))
        except Exception:  # noqa: BLE001 - 判定失败按未完成处理
            logger.warning("最终结果判定失败，按任务失败处理", exc_info=True)
            return False
    
    def _generate_reasoning(self, tool_correctness: float, 
                           steps_correctness: float, 
                           task_success: bool) -> str:
        """生成评估理由"""
        parts = []
        
        parts.append(f"工具使用正确性: {tool_correctness:.2f}")
        parts.append(f"步骤正确性: {steps_correctness:.2f}")
        parts.append(f"任务成功: {'是' if task_success else '否'}")
        
        return "; ".join(parts)
    
    def get_dimension_name(self) -> str:
        return "task_completion"
