"""评估任务编排服务。

后台任务自带独立 Session（见 ``app.tasks.evaluation_task``）——
请求级的 Session 不是线程安全的，不能跨线程复用。
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.evaluators.performance import PerformanceEvaluator
from app.evaluators.quality import QualityEvaluator
from app.evaluators.security import SecurityEvaluator
from app.evaluators.task_completion import TaskCompletionEvaluator
from app.models.datasets import Dataset, TestCase
from app.models.evaluation import Evaluation, EvaluationMetric
from app.models.execution import AgentExecution
from app.models.security import SecurityLog, SecurityRule
from app.monitoring.cost_calculator import CostCalculator
from app.monitoring.metrics import MetricsCollector

logger = logging.getLogger(__name__)

# 某个维度评估崩溃时的兜底结果，避免一个维度失败拖垮整条用例
_QUALITY_FALLBACK: Dict[str, Any] = {
    "score": 0.0,
    "metrics": {
        "accuracy": 0.0,
        "completeness": 0.0,
        "hallucination_rate": 1.0,
        "relevance": 0.0,
    },
    "reasoning": "质量评估异常，按最差情况计分",
}
_PERFORMANCE_FALLBACK: Dict[str, Any] = {
    "score": 0.0,
    "metrics": {
        "latency": 0.0,
        "throughput": 0.0,
        "token_efficiency": 0.0,
        "cost": 0.0,
    },
    "reasoning": "性能评估异常，按最差情况计分",
}
_SECURITY_FALLBACK: Dict[str, Any] = {
    "score": 0.0,
    "metrics": {
        "prompt_leakage": False,
        "dangerous_operation": False,
        "privilege_escalation": False,
        "rule_violations": [],
        "llm_findings": {},
    },
    "reasoning": "安全评估异常，按最差情况计分",
}
_TASK_FALLBACK: Dict[str, Any] = {
    "score": 0.0,
    "metrics": {
        "task_success": False,
        "steps_correctness": 0.0,
        "tool_usage_correctness": 0.0,
    },
    "reasoning": "任务完成度评估异常，按最差情况计分",
}


def _with_reason(fallback: Dict[str, Any], reason: str) -> Dict[str, Any]:
    result = {
        "score": fallback["score"],
        "metrics": dict(fallback["metrics"]),
        "reasoning": f"{fallback['reasoning']}（{reason}）",
    }
    return result


class EvaluationService:
    """评估服务"""

    def __init__(self, db: Session, config: Optional[Dict[str, Any]] = None,
                 settings: Optional[Settings] = None) -> None:
        self.db = db
        self.config = config or {}
        self.settings = settings or get_settings()

        evaluator_config: Dict[str, Any] = dict(self.config.get("evaluators", {}))
        evaluator_config.setdefault("model", self.settings.LLM_MODEL)
        if self.settings.LLM_BASE_URL:
            evaluator_config.setdefault("base_url", self.settings.LLM_BASE_URL)

        # 让 security_rules 表真正生效：加载库中启用的规则注入规则引擎
        db_rules = self._load_security_rules()
        security_config = dict(evaluator_config)
        if db_rules:
            existing = list(security_config.get("custom_rules", []) or [])
            security_config["custom_rules"] = existing + db_rules

        self.quality_evaluator = QualityEvaluator(evaluator_config)
        self.performance_evaluator = PerformanceEvaluator(evaluator_config)
        self.security_evaluator = SecurityEvaluator(security_config)
        self.task_evaluator = TaskCompletionEvaluator(evaluator_config)

        self.cost_calculator = CostCalculator()
        self.metrics_collector = MetricsCollector(db)

    def _load_security_rules(self) -> list:
        """读取数据库中启用的安全规则，转换为规则引擎所需的结构。"""
        try:
            rows = self.db.query(SecurityRule).filter(SecurityRule.enabled.is_(True)).all()
        except Exception as exc:  # noqa: BLE001 - 表尚未创建时不影响评估主流程
            logger.warning("读取安全规则失败，仅使用内置规则: %s", exc)
            self.db.rollback()
            return []

        rules = []
        for row in rows:
            if not row.pattern:
                continue
            rules.append(
                {
                    "name": row.name,
                    "pattern": row.pattern,
                    "type": row.rule_type or "custom",
                    "severity": row.severity or "medium",
                    "description": row.description or f"命中自定义规则: {row.name}",
                }
            )
        return rules

    # ------------------------------------------------------------------ 创建
    def create_evaluation(self, dataset_id: int, agent_config: Dict[str, Any]) -> Evaluation:
        """创建评估任务记录"""
        dataset = self.db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} not found")

        evaluation = Evaluation(
            dataset_id=dataset_id,
            agent_name=str(agent_config.get("name", "unknown")),
            agent_version=str(agent_config.get("version", "1.0")),
            agent_config=agent_config,
            status="running",
            started_at=datetime.utcnow(),
        )

        self.db.add(evaluation)
        self.db.commit()
        self.db.refresh(evaluation)
        return evaluation

    # ------------------------------------------------------------------ 运行
    def run_evaluation(self, evaluation_id: int, agent_chain: Any,
                       case_limit: Optional[int] = None) -> Dict[str, Any]:
        """同步运行评估，遍历数据集中的测试用例。

        ``case_limit`` 限制只跑前 N 条（网页端用于快速试跑），None 表示全部。
        """
        evaluation = self.db.query(Evaluation).filter(Evaluation.id == evaluation_id).first()
        if not evaluation:
            raise ValueError(f"Evaluation {evaluation_id} not found")

        from app.agents.executor import TrackedAgentExecutor

        dataset = evaluation.dataset
        test_cases = list(dataset.test_cases or [])
        if case_limit and case_limit > 0:
            test_cases = test_cases[:int(case_limit)]

        model_name = getattr(agent_chain, "model_name", None) or self.settings.LLM_MODEL
        executor = TrackedAgentExecutor(agent_chain, model_name=model_name)

        results: Dict[str, Any] = {
            "total_cases": len(test_cases),
            "successful": 0,
            "failed": 0,
            "dimension_scores": {
                "quality": [],
                "performance": [],
                "security": [],
                "task_completion": [],
            },
            "executions": [],
            "total_cost": 0.0,
            "total_tokens": 0,
        }

        from concurrent.futures import ThreadPoolExecutor

        # 四维度裁判并发执行，整轮复用同一个线程池
        with ThreadPoolExecutor(max_workers=4) as pool:
            for test_case in test_cases:
                self._run_single_case(evaluation, test_case, executor, model_name, results, pool)
                self._flush_progress(evaluation, results, len(test_cases))

        evaluation = self._calculate_final_scores(evaluation, results)
        detailed_metrics = self.metrics_collector.collect_evaluation_metrics(evaluation_id)

        evaluation.completed_at = datetime.utcnow()
        evaluation.status = self._resolve_status(results)
        evaluation.metrics = detailed_metrics
        evaluation.total_cost = results["total_cost"]
        evaluation.total_tokens = results["total_tokens"]

        try:
            self.db.commit()
        except Exception:  # noqa: BLE001 - 提交失败需回滚并向上抛出
            self.db.rollback()
            raise

        logger.info(
            "评估 %s 完成: 成功 %s / 失败 %s, 总分 %.2f",
            evaluation_id,
            results["successful"],
            results["failed"],
            evaluation.overall_score or 0.0,
        )

        return {
            "evaluation_id": evaluation_id,
            "overall_score": evaluation.overall_score,
            "dimension_scores": {
                "quality": evaluation.quality_score,
                "performance": evaluation.performance_score,
                "security": evaluation.security_score,
                "task_completion": evaluation.task_completion_score,
            },
            "detailed_metrics": detailed_metrics,
            "executions": results["executions"],
        }

    def _flush_progress(self, evaluation: Evaluation, results: Dict[str, Any], total: int) -> None:
        """每跑完一条就写回计数，让前端轮询能看到实时进度。

        单条用例失败时 ``_run_single_case`` 内部已 rollback，这里重新赋值再提交即可。
        """
        try:
            evaluation.total_cases = total
            evaluation.successful_cases = results["successful"]
            evaluation.failed_cases = results["failed"]
            self.db.commit()
        except Exception as exc:  # noqa: BLE001 - 进度写回失败不影响评估主流程
            logger.warning("写入评估进度失败: %s", exc)
            self.db.rollback()

    def _run_single_case(self, evaluation: Evaluation, test_case: TestCase,
                         executor: Any, model_name: str, results: Dict[str, Any],
                         pool: Any = None) -> None:
        try:
            exec_result = executor.execute(test_case.input_prompt)
            summary = exec_result.get("execution_summary") or {}

            execution = AgentExecution(
                test_case_id=test_case.id,
                evaluation_id=evaluation.id,
                agent_name=evaluation.agent_name,
                agent_version=evaluation.agent_version,
                status=exec_result.get("status", "failed"),
                final_response=exec_result.get("final_response", ""),
                completed_at=datetime.utcnow(),
                total_duration=summary.get("total_duration", 0.0),
                total_tokens=summary.get("total_tokens", 0),
                prompt_tokens=summary.get("prompt_tokens", 0),
                completion_tokens=summary.get("completion_tokens", 0),
                estimated_cost=exec_result.get("estimated_cost", 0.0),
                steps=summary.get("steps", []),
                tool_calls=summary.get("tool_calls", []),
                intermediate_outputs=summary.get("intermediate_outputs", []),
                error_message=exec_result.get("error"),
                execution_metadata={
                    "model_name": model_name,
                    "token_source": summary.get("token_source"),
                    "latency_source": summary.get("latency_source"),
                },
            )

            self.db.add(execution)
            self.db.flush()

            results["total_cost"] += exec_result.get("estimated_cost", 0.0) or 0.0
            results["total_tokens"] += summary.get("total_tokens", 0) or 0

            if exec_result.get("status") != "completed":
                results["failed"] += 1
                results["executions"].append(
                    {
                        "test_case_id": test_case.id,
                        "status": "failed",
                        "error": exec_result.get("error"),
                    }
                )
                self.db.commit()
                return

            scored = self._evaluate_dimensions(execution, test_case, pool)
            quality_result = scored["quality"]
            performance_result = scored["performance"]
            security_result = scored["security"]
            task_result = scored["task_completion"]

            metric = EvaluationMetric(
                evaluation_id=evaluation.id,
                execution_id=execution.id,
                accuracy=quality_result["metrics"]["accuracy"],
                completeness=quality_result["metrics"]["completeness"],
                hallucination_rate=quality_result["metrics"]["hallucination_rate"],
                relevance=quality_result["metrics"]["relevance"],
                quality_reasoning=quality_result["reasoning"],
                latency=performance_result["metrics"]["latency"],
                throughput=performance_result["metrics"]["throughput"],
                token_efficiency=performance_result["metrics"]["token_efficiency"],
                cost=performance_result["metrics"]["cost"],
                prompt_leakage=bool(security_result["metrics"]["prompt_leakage"]),
                dangerous_operation=bool(security_result["metrics"]["dangerous_operation"]),
                privilege_escalation=bool(security_result["metrics"]["privilege_escalation"]),
                security_reasoning=security_result["reasoning"],
                task_success=bool(task_result["metrics"]["task_success"]),
                steps_correctness=task_result["metrics"]["steps_correctness"],
                tool_usage_correctness=task_result["metrics"]["tool_usage_correctness"],
                task_reasoning=task_result["reasoning"],
            )
            self.db.add(metric)
            self._record_security_logs(execution, security_result)

            for dimension, result in (
                ("quality", quality_result),
                ("performance", performance_result),
                ("security", security_result),
                ("task_completion", task_result),
            ):
                results["dimension_scores"][dimension].append(result["score"])

            results["successful"] += 1
            results["executions"].append(
                {
                    "test_case_id": test_case.id,
                    "status": "completed",
                    "quality_score": quality_result["score"],
                    "performance_score": performance_result["score"],
                    "security_score": security_result["score"],
                    "task_score": task_result["score"],
                }
            )
            self.db.commit()

        except Exception as exc:  # noqa: BLE001 - 单条用例失败不影响整批
            self.db.rollback()
            results["failed"] += 1
            logger.exception("测试用例 %s 评估失败: %s", test_case.id, exc)
            results["executions"].append(
                {"test_case_id": test_case.id, "status": "failed", "error": str(exc)}
            )

    # 维度名 -> (评估器属性名, 兜底结果)
    _DIMENSIONS = (
        ("quality", "quality_evaluator", _QUALITY_FALLBACK),
        ("performance", "performance_evaluator", _PERFORMANCE_FALLBACK),
        ("security", "security_evaluator", _SECURITY_FALLBACK),
        ("task_completion", "task_evaluator", _TASK_FALLBACK),
    )

    def _evaluate_dimensions(self, execution: AgentExecution, test_case: TestCase,
                             pool: Any = None) -> Dict[str, Dict[str, Any]]:
        """四维度打分。

        四个评估器只做 LLM 调用、不碰数据库，因此可以用线程池并发——
        裁判调用是纯 I/O 等待，串行时它占了单条用例七成的耗时。
        ``pool`` 为 None 时退化为串行。
        """
        jobs = [
            (dimension, getattr(self, attribute), fallback)
            for dimension, attribute, fallback in self._DIMENSIONS
        ]

        if pool is None:
            return {
                dimension: self._safe_evaluate(evaluator, execution, test_case, fallback)
                for dimension, evaluator, fallback in jobs
            }

        futures = {
            dimension: pool.submit(self._safe_evaluate, evaluator, execution, test_case, fallback)
            for dimension, evaluator, fallback in jobs
        }
        return {dimension: future.result() for dimension, future in futures.items()}

    @staticmethod
    def _safe_evaluate(evaluator: Any, execution: AgentExecution,
                       test_case: TestCase, fallback: Dict[str, Any]) -> Dict[str, Any]:
        """单个评估维度失败时使用兜底结果，不中断整条用例。"""
        try:
            result = evaluator.evaluate(execution, test_case)
            if isinstance(result, dict) and "metrics" in result:
                return result
            return _with_reason(fallback, "返回结构不合法")
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s 评估失败: %s", evaluator.__class__.__name__, exc)
            return _with_reason(fallback, str(exc))

    # ------------------------------------------------------------------ 汇总
    @staticmethod
    def _resolve_status(results: Dict[str, Any]) -> str:
        total = results["total_cases"]
        successful = results["successful"]
        if total and successful == 0:
            return "failed"
        if successful and successful < total:
            return "partial"
        if total == 0:
            return "failed"
        return "completed"

    def _calculate_final_scores(self, evaluation: Evaluation, results: Dict[str, Any]) -> Evaluation:
        """按配置权重计算各维度均分与综合得分（0-100）"""

        def mean(values: list) -> float:
            return (sum(values) / len(values)) if values else 0.0

        evaluation.quality_score = mean(results["dimension_scores"]["quality"]) * 100
        evaluation.performance_score = mean(results["dimension_scores"]["performance"]) * 100
        evaluation.security_score = mean(results["dimension_scores"]["security"]) * 100
        evaluation.task_completion_score = mean(results["dimension_scores"]["task_completion"]) * 100

        evaluation.overall_score = (
            evaluation.quality_score * self.settings.WEIGHT_QUALITY
            + evaluation.performance_score * self.settings.WEIGHT_PERFORMANCE
            + evaluation.security_score * self.settings.WEIGHT_SECURITY
            + evaluation.task_completion_score * self.settings.WEIGHT_TASK_COMPLETION
        )

        evaluation.total_cases = results["total_cases"]
        evaluation.successful_cases = results["successful"]
        evaluation.failed_cases = results["failed"]
        return evaluation

    def _record_security_logs(self, execution: AgentExecution, security_result: Dict[str, Any]) -> None:
        """记录规则引擎命中的安全日志"""
        rule_violations = security_result.get("metrics", {}).get("rule_violations", []) or []

        for violation in rule_violations:
            log = SecurityLog(
                execution_id=execution.id,
                event_type=violation.get("type", "other"),
                severity=violation.get("severity", "medium"),
                description=violation.get("description", ""),
                detection_method="rule",
                context=violation,
                triggered_by=violation.get("rule_name", "auto_detection"),
                status="detected",
            )
            self.db.add(log)
