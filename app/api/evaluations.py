"""评估任务接口。"""

from typing import Any, Dict, List

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.evaluation import Evaluation
from app.models.execution import AgentExecution
from app.schemas.evaluation import (
    EvaluationCreate,
    EvaluationDetailResponse,
    EvaluationResponse,
    ExecutionResponse,
)
from app.services.evaluation_service import EvaluationService
from app.tasks.evaluation_task import run_evaluation_job

router = APIRouter(prefix="/evaluations", tags=["evaluations"])


@router.post("/", response_model=EvaluationResponse, status_code=201)
def create_evaluation(
    request: EvaluationCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """创建评估任务并在后台异步执行。

    后台任务自带 Session，不复用此处的请求级 Session。
    """
    service = EvaluationService(db)

    try:
        evaluation = service.create_evaluation(
            dataset_id=request.dataset_id,
            agent_config=request.agent_config,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    background_tasks.add_task(run_evaluation_job, evaluation.id, request.agent_config)

    return EvaluationResponse.model_validate(evaluation)


@router.get("/", response_model=List[EvaluationResponse])
def list_evaluations(
    dataset_id: int | None = None,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    """分页查询评估任务列表。"""
    query = db.query(Evaluation)
    if dataset_id is not None:
        query = query.filter(Evaluation.dataset_id == dataset_id)

    evaluations = (
        query.order_by(Evaluation.id.desc()).offset(skip).limit(limit).all()
    )
    return [EvaluationResponse.model_validate(item) for item in evaluations]


@router.get("/{evaluation_id}", response_model=EvaluationDetailResponse)
def get_evaluation(evaluation_id: int, db: Session = Depends(get_db)):
    """获取评估详情（含四维度得分与聚合指标）。"""
    evaluation = db.query(Evaluation).filter(Evaluation.id == evaluation_id).first()
    if not evaluation:
        raise HTTPException(status_code=404, detail="Evaluation not found")

    return EvaluationDetailResponse.model_validate(evaluation)


@router.get("/{evaluation_id}/executions", response_model=List[ExecutionResponse])
def get_evaluation_executions(evaluation_id: int, db: Session = Depends(get_db)):
    """获取一次评估中每条用例的执行轨迹。"""
    executions = (
        db.query(AgentExecution)
        .filter(AgentExecution.evaluation_id == evaluation_id)
        .order_by(AgentExecution.id)
        .all()
    )
    return [ExecutionResponse.model_validate(item) for item in executions]


@router.get("/{evaluation_id}/metrics")
def get_evaluation_metrics(evaluation_id: int, db: Session = Depends(get_db)) -> Dict[str, Any]:
    """获取评估的聚合指标（延迟分位、Token、成本、吞吐量）。"""
    from app.monitoring.metrics import MetricsCollector

    evaluation = db.query(Evaluation).filter(Evaluation.id == evaluation_id).first()
    if not evaluation:
        raise HTTPException(status_code=404, detail="Evaluation not found")

    return MetricsCollector(db).collect_evaluation_metrics(evaluation_id)
