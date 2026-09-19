"""数据集与测试用例的请求 / 响应模型。"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, field_validator

from app.core.config import get_settings


class TestCaseCreate(BaseModel):
    input_prompt: str
    ground_truth: str
    expected_tools: List[str] = []
    expected_steps: List[Any] = []
    quality_weight: float = 1.0
    performance_weight: float = 1.0
    security_weight: float = 1.0
    task_weight: float = 1.0
    security_rules: Dict[str, Any] = {}
    difficulty: Optional[str] = None
    tags: List[str] = []

    @field_validator("difficulty")
    @classmethod
    def _check_difficulty(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        allowed = {"easy", "medium", "hard"}
        normalized = value.strip().lower()
        if normalized not in allowed:
            raise ValueError(f"difficulty 必须是 {sorted(allowed)} 之一")
        return normalized


class TestCaseResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    dataset_id: int
    input_prompt: str
    ground_truth: str
    expected_tools: Optional[List[str]] = None
    expected_steps: Optional[List[Any]] = None
    difficulty: Optional[str] = None
    tags: Optional[List[str]] = None
    created_at: Optional[datetime] = None


class DatasetCreate(BaseModel):
    name: str
    description: Optional[str] = None
    category: Optional[str] = None
    config: Dict[str, Any] = {}
    test_cases: List[TestCaseCreate] = []


class DatasetResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    description: Optional[str] = None
    category: Optional[str] = None
    created_at: Optional[datetime] = None
    test_case_count: int = 0


class DatasetDetailResponse(DatasetResponse):
    config: Optional[Dict[str, Any]] = None
    test_cases: List[TestCaseResponse] = []


def embedding_dimension() -> int:
    """对外暴露向量维度，便于文档与校验保持一致。"""
    return get_settings().EMBEDDING_DIMENSION
