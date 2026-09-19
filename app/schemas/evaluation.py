"""评估任务相关的请求 / 响应模型。"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class EvaluationCreate(BaseModel):
    """创建评估任务。"""

    dataset_id: int = Field(..., description="目标数据集 ID")
    agent_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="被测 Agent 配置（模型、系统提示词、工具等）",
    )


class EvaluationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    dataset_id: int
    agent_name: str
    status: str
    started_at: Optional[datetime] = None


class ExecutionResponse(BaseModel):
    """单次 Agent 执行记录。"""

    model_config = ConfigDict(from_attributes=True)

    id: int
    test_case_id: int
    status: str
    final_response: Optional[str] = None
    total_duration: Optional[float] = None
    total_tokens: Optional[int] = None
    estimated_cost: Optional[float] = None
    steps: Optional[List[Any]] = None
    tool_calls: Optional[List[Any]] = None
    error_message: Optional[str] = None


class EvaluationDetailResponse(BaseModel):
    """评估任务详情，含四个维度得分与聚合指标。"""

    model_config = ConfigDict(from_attributes=True)

    id: int
    dataset_id: int
    agent_name: str
    status: str
    overall_score: Optional[float] = None
    quality_score: Optional[float] = None
    performance_score: Optional[float] = None
    security_score: Optional[float] = None
    task_completion_score: Optional[float] = None
    metrics: Optional[Dict[str, Any]] = None
    total_cases: Optional[int] = None
    successful_cases: Optional[int] = None
    failed_cases: Optional[int] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
