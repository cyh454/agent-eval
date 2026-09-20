"""安全规则管理与安全日志查询接口。

这里的规则会被 EvaluationService 注入 SecurityDetector，参与安全维度打分。
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.execution import AgentExecution
from app.models.security import SecurityLog, SecurityRule
from app.schemas.security import (
    SecurityLogResponse,
    SecurityRuleCreate,
    SecurityRuleResponse,
    SecurityRuleUpdate,
)

router = APIRouter(prefix="/security", tags=["security"])


@router.get("/rules", response_model=List[SecurityRuleResponse])
def list_rules(enabled: Optional[bool] = None, db: Session = Depends(get_db)):
    """列出安全规则。"""
    query = db.query(SecurityRule)
    if enabled is not None:
        query = query.filter(SecurityRule.enabled.is_(bool(enabled)))
    return [SecurityRuleResponse.model_validate(r) for r in query.order_by(SecurityRule.id).all()]


@router.post("/rules", response_model=SecurityRuleResponse, status_code=201)
def create_rule(request: SecurityRuleCreate, db: Session = Depends(get_db)):
    """新增安全规则。"""
    exists = db.query(SecurityRule).filter(SecurityRule.name == request.name).first()
    if exists:
        raise HTTPException(status_code=409, detail=f"Rule '{request.name}' already exists")

    rule = SecurityRule(**request.model_dump())
    db.add(rule)
    db.commit()
    db.refresh(rule)
    return SecurityRuleResponse.model_validate(rule)


@router.patch("/rules/{rule_id}", response_model=SecurityRuleResponse)
def update_rule(rule_id: int, request: SecurityRuleUpdate, db: Session = Depends(get_db)):
    """局部更新安全规则。"""
    rule = db.query(SecurityRule).filter(SecurityRule.id == rule_id).first()
    if not rule:
        raise HTTPException(status_code=404, detail="Rule not found")

    for field, value in request.model_dump(exclude_unset=True).items():
        setattr(rule, field, value)

    db.commit()
    db.refresh(rule)
    return SecurityRuleResponse.model_validate(rule)


@router.delete("/rules/{rule_id}", status_code=204)
def delete_rule(rule_id: int, db: Session = Depends(get_db)):
    """删除安全规则。"""
    rule = db.query(SecurityRule).filter(SecurityRule.id == rule_id).first()
    if not rule:
        raise HTTPException(status_code=404, detail="Rule not found")

    db.delete(rule)
    db.commit()
    return None


@router.get("/logs", response_model=List[SecurityLogResponse])
def list_security_logs(
    evaluation_id: Optional[int] = None,
    severity: Optional[str] = None,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db),
):
    """查询安全事件日志，可按评估任务或严重级别过滤。"""
    query = db.query(SecurityLog)

    if evaluation_id is not None:
        query = query.join(AgentExecution, SecurityLog.execution_id == AgentExecution.id).filter(
            AgentExecution.evaluation_id == evaluation_id
        )
    if severity:
        query = query.filter(SecurityLog.severity == severity)

    logs = query.order_by(SecurityLog.id.desc()).offset(skip).limit(limit).all()
    return [SecurityLogResponse.model_validate(log) for log in logs]
