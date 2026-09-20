"""后台评估任务。

每次任务自带独立 Session：请求作用域的 Session 不是线程安全的，
且请求返回后连接可能已被回收。
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional

from app.agents.builder import AgentBuilder
from app.database import SessionLocal
from app.models.evaluation import Evaluation
from app.services.evaluation_service import EvaluationService

logger = logging.getLogger(__name__)


def run_evaluation_job(evaluation_id: int, agent_config: Dict[str, Any],
                       case_limit: Optional[int] = None) -> None:
    """构建被测 Agent 并运行评估，全程使用独立 Session。

    ``case_limit`` 只跑前 N 条用例（网页端快速试跑），None 表示全部。
    """
    db = SessionLocal()
    try:
        agent_chain = AgentBuilder().build(agent_config)
        EvaluationService(db).run_evaluation(evaluation_id, agent_chain, case_limit=case_limit)
    except Exception as exc:  # noqa: BLE001 - 后台任务异常必须落库，不能静默丢失
        logger.exception("评估任务 %s 执行失败: %s", evaluation_id, exc)
        db.rollback()
        evaluation = db.query(Evaluation).filter(Evaluation.id == evaluation_id).first()
        if evaluation is not None:
            evaluation.status = "failed"
            evaluation.completed_at = datetime.utcnow()
            db.commit()
    finally:
        db.close()
