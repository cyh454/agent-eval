"""安全规则与安全日志的请求 / 响应模型。"""

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict


class SecurityRuleCreate(BaseModel):
    name: str
    category: Optional[str] = None
    rule_type: str = "keyword"
    pattern: str
    threshold: Optional[float] = None
    enabled: bool = True
    severity: str = "medium"
    description: Optional[str] = None


class SecurityRuleUpdate(BaseModel):
    category: Optional[str] = None
    rule_type: Optional[str] = None
    pattern: Optional[str] = None
    threshold: Optional[float] = None
    enabled: Optional[bool] = None
    severity: Optional[str] = None
    description: Optional[str] = None


class SecurityRuleResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    category: Optional[str] = None
    rule_type: Optional[str] = None
    pattern: Optional[str] = None
    severity: Optional[str] = None
    enabled: Optional[bool] = None
    description: Optional[str] = None
    created_at: Optional[datetime] = None


class SecurityLogResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    execution_id: int
    event_type: Optional[str] = None
    severity: Optional[str] = None
    description: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    triggered_by: Optional[str] = None
    detected_at: Optional[datetime] = None
