from typing import Dict, List, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from app.models.execution import AgentExecution
from app.models.evaluation import Evaluation

class MetricsCollector:
    """指标收集器"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def collect_evaluation_metrics(self, evaluation_id: int) -> Dict:
        """收集评估指标"""
        
        evaluation = self.db.query(Evaluation).filter(
            Evaluation.id == evaluation_id
        ).first()
        
        if not evaluation:
            return {}
        
        executions = self.db.query(AgentExecution).filter(
            AgentExecution.evaluation_id == evaluation_id
        ).all()
        
        # 基础统计
        total_executions = len(executions)
        successful = sum(1 for e in executions if e.status == "completed")
        failed = sum(1 for e in executions if e.status == "failed")
        
        # 性能统计
        latencies = [e.total_duration for e in executions if e.total_duration]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        max_latency = max(latencies) if latencies else 0
        min_latency = min(latencies) if latencies else 0
        
        # Token 统计（字段可能为 None，统一兜底为 0）
        total_tokens = sum(e.total_tokens or 0 for e in executions)
        total_prompt_tokens = sum(e.prompt_tokens or 0 for e in executions)
        total_completion_tokens = sum(e.completion_tokens or 0 for e in executions)
        
        # 成本统计
        total_cost = sum(e.estimated_cost or 0 for e in executions)
        
        # 计算吞吐量（假设评估运行时间）
        if evaluation.started_at and evaluation.completed_at:
            duration = (evaluation.completed_at - evaluation.started_at).total_seconds()
            throughput = (total_executions / duration * 60) if duration > 0 else 0
        else:
            throughput = 0
        
        return {
            "evaluation_id": evaluation_id,
            "total_executions": total_executions,
            "successful": successful,
            "failed": failed,
            "success_rate": successful / total_executions if total_executions > 0 else 0,
            
            "performance": {
                "avg_latency": avg_latency,
                "max_latency": max_latency,
                "min_latency": min_latency,
                "throughput": throughput,
                "p50_latency": self._percentile(latencies, 50),
                "p95_latency": self._percentile(latencies, 95),
                "p99_latency": self._percentile(latencies, 99),
            },
            
            "tokens": {
                "total": total_tokens,
                "prompt": total_prompt_tokens,
                "completion": total_completion_tokens,
                "avg_per_execution": total_tokens / total_executions if total_executions > 0 else 0,
            },
            
            "cost": {
                "total": total_cost,
                "avg_per_execution": total_cost / total_executions if total_executions > 0 else 0,
            }
        }
    
    def _percentile(self, data: List[float], p: int) -> float:
        """计算百分位数"""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * p / 100
        f = int(k)
        c = f + 1 if f + 1 < len(sorted_data) else f
        if f == c:
            return sorted_data[f]
        return sorted_data[f] * (c - k) + sorted_data[c] * (k - f)
