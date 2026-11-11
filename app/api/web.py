"""网页版一键评估接口。

给 UI 用的三个动作：

1. ``POST /web/probe``    探测目标服务的对话端点（只给根地址即可）
2. ``POST /web/evaluate`` 发起评估，后台异步跑
3. ``GET  /web/result/{id}`` 轮询结果（含进度、四维度得分、逐条明细）
"""

import hashlib
import html
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.agents.probe import DEFAULT_PROBE, probe_agent_service
from app.core.config import get_settings
from app.database import get_db
from app.models.datasets import Dataset, TestCase
from app.models.evaluation import Evaluation, EvaluationMetric
from app.models.execution import AgentExecution
from app.services.evaluation_service import EvaluationService
from app.tasks.evaluation_task import run_evaluation_job

router = APIRouter(prefix="/web", tags=["web"])


class ProbeRequest(BaseModel):
    base_url: str = Field(..., description="被测 Agent 服务根地址，如 http://localhost:8000")


class EvaluateRequest(BaseModel):
    base_url: str
    dataset_id: int
    case_limit: int = Field(10, ge=1, le=200, description="只跑前 N 条用例，0 或省略表示全部")
    run_all: bool = False
    # 探测失败或需要指定时，手工覆盖下面任一字段
    chat_path: Optional[str] = None
    request_field: Optional[str] = None
    reply_field: Optional[str] = None
    tool_calls_field: Optional[str] = None
    model: Optional[str] = None
    agent_name: Optional[str] = None


def _default_dataset_id(db: Session, dataset_id: Optional[int]) -> int:
    """未指定数据集时，取用例数最多的那个（通常是刚导入的评测集）。"""
    from sqlalchemy import func

    if dataset_id:
        return dataset_id
    # 只在「有用例」的数据集里挑，避免选中空数据集（如历史脏数据）白跑一场
    row = (
        db.query(TestCase.dataset_id, func.count(TestCase.id).label("n"))
        .group_by(TestCase.dataset_id)
        .having(func.count(TestCase.id) > 0)
        .order_by(func.count(TestCase.id).desc())
        .first()
    )
    if row is None:
        raise HTTPException(status_code=400, detail="还没有任何带用例的数据集，请先导入测试用例")
    return row[0]


@router.post("/probe")
def probe(request: ProbeRequest) -> Dict[str, Any]:
    """探测服务地址，返回自动识别到的对话端点与字段映射。"""
    return probe_agent_service(request.base_url)


@router.post("/evaluate", status_code=201)
def evaluate(
    request: EvaluateRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """发起一次评估：探测 → 组装 adapter 配置 → 后台跑。"""
    settings = get_settings()

    probe_result = probe_agent_service(request.base_url)

    # 自指必须硬拦：探测失败原本会兜底继续跑，那条退路会让「评自己」静默通过
    if probe_result.get("self_reference"):
        raise HTTPException(
            status_code=400,
            detail=f"目标地址解析到 agent-eval 自身（{probe_result.get('error')}），"
                   f"请填入被测 Agent 的服务地址",
        )

    if not probe_result.get("ok") and not request.chat_path:
        # 探测失败且没手工指定，仍按兜底配置试一把，但把警告带回前端
        probe_result["warning"] = (
            f"自动探测失败（{probe_result.get('error')}），已按默认端点尝试，可在高级选项里手工指定"
        )

    def pick(key: str) -> str:
        return str(request.model_dump().get(key) or probe_result.get(key) or DEFAULT_PROBE[key])

    agent_config: Dict[str, Any] = {
        "type": "remote_http",
        "base_url": probe_result.get("base_url") or request.base_url,
        "chat_path": pick("chat_path"),
        "request_field": pick("request_field"),
        "reply_field": pick("reply_field"),
        "tool_calls_field": pick("tool_calls_field"),
        "model": request.model or settings.LLM_MODEL,
        "name": request.agent_name or probe_result.get("service_title") or "remote-agent",
        "version": "1.0",
        "timeout": 300,
    }

    dataset_id = _default_dataset_id(db, request.dataset_id)

    # 空数据集直接拒绝，而不是让它跑到「0 条用例 → failed」才暴露
    case_count = db.query(TestCase).filter(TestCase.dataset_id == dataset_id).count()
    if case_count == 0:
        raise HTTPException(
            status_code=400,
            detail=f"数据集 #{dataset_id} 没有任何测试用例，请换一个数据集或先导入用例",
        )

    service = EvaluationService(db)
    try:
        evaluation = service.create_evaluation(dataset_id, agent_config)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    limit = None if request.run_all else min(request.case_limit, case_count)
    background_tasks.add_task(run_evaluation_job, evaluation.id, agent_config, limit)

    return {
        "evaluation_id": evaluation.id,
        "dataset_id": dataset_id,
        "case_limit": limit,
        "probe": probe_result,
        "agent_config": agent_config,
    }


def _applied_dimensions(evaluation: Evaluation) -> List[str]:
    """哪些维度参与了总分合成。

    读路径上直接从评估记录反推：分数为 NULL 表示该维度用例权重为 0（不适用），
    这样不必依赖写入时落盘的中间结果，历史数据也一致。
    """
    fields = (
        ("quality", evaluation.quality_score),
        ("performance", evaluation.performance_score),
        ("security", evaluation.security_score),
        ("task_completion", evaluation.task_completion_score),
    )
    return [name for name, value in fields if value is not None]


def _collect_executions(db: Session, evaluation_id: int) -> List[Dict[str, Any]]:
    """某次评估的逐条明细：输入、参考答案、工具轨迹、裁判理由、原始指标。"""
    rows = (
        db.query(AgentExecution, TestCase.input_prompt, TestCase.ground_truth, EvaluationMetric)
        .outerjoin(TestCase, TestCase.id == AgentExecution.test_case_id)
        .outerjoin(EvaluationMetric, EvaluationMetric.execution_id == AgentExecution.id)
        .filter(AgentExecution.evaluation_id == evaluation_id)
        .order_by(AgentExecution.id)
        .all()
    )

    items: List[Dict[str, Any]] = []
    for execution, input_prompt, ground_truth, metric in rows:
        calls = [c for c in (execution.tool_calls or []) if isinstance(c, dict)]
        items.append({
            "id": execution.id,
            "input_prompt": input_prompt or "",
            "ground_truth": ground_truth or "",
            "status": execution.status,
            "duration": execution.total_duration,
            "tokens": execution.total_tokens,
            "cost": execution.estimated_cost,
            "tool_call_count": len(calls),
            "tool_names": [c.get("name") for c in calls],
            "tools": [
                {
                    "name": c.get("name") or "unknown",
                    "arguments": str(c.get("arguments") or "")[:300],
                    "output": str(c.get("output") or "")[:400],
                    "status": c.get("status") or "completed",
                }
                for c in calls
            ],
            "response": (execution.final_response or "")[:1200],
            "error": execution.error_message,
            # 数据口径：告诉读数的人这些数是测出来的、报上来的，还是估出来的
            "sources": {
                "token": ((execution.execution_metadata or {}).get("token_source") or "missing"),
                "latency": ((execution.execution_metadata or {}).get("latency_source") or "client"),
            },
            "reasoning": {
                "quality": getattr(metric, "quality_reasoning", None),
                "security": getattr(metric, "security_reasoning", None),
                "task": getattr(metric, "task_reasoning", None),
            } if metric is not None else {},
            "indicators": {
                "security_score": getattr(metric, "security_score", None),
                "accuracy": getattr(metric, "accuracy", None),
                "completeness": getattr(metric, "completeness", None),
                "relevance": getattr(metric, "relevance", None),
                "hallucination_rate": getattr(metric, "hallucination_rate", None),
                "task_success": getattr(metric, "task_success", None),
                "steps_correctness": getattr(metric, "steps_correctness", None),
                "tool_usage_correctness": getattr(metric, "tool_usage_correctness", None),
                "prompt_leakage": getattr(metric, "prompt_leakage", None),
                "dangerous_operation": getattr(metric, "dangerous_operation", None),
                "privilege_escalation": getattr(metric, "privilege_escalation", None),
            } if metric is not None else {},
        })
    return items


def _summary(evaluation: Evaluation) -> Dict[str, Any]:
    """评估概要，对比与列表复用。"""
    return {
        "id": evaluation.id,
        "agent_name": evaluation.agent_name,
        "dataset_id": evaluation.dataset_id,
        "status": evaluation.status,
        "started_at": evaluation.started_at.isoformat() if evaluation.started_at else None,
        "completed_at": evaluation.completed_at.isoformat() if evaluation.completed_at else None,
        "progress": {
            "total": evaluation.total_cases,
            "successful": evaluation.successful_cases,
            "failed": evaluation.failed_cases,
        },
        "scores": {
            "overall": evaluation.overall_score,
            "quality": evaluation.quality_score,
            "performance": evaluation.performance_score,
            "security": evaluation.security_score,
            "task_completion": evaluation.task_completion_score,
        },
        "totals": {
            "cost": evaluation.total_cost,
            "tokens": evaluation.total_tokens,
        },
    }


@router.get("/result/{evaluation_id}")
def get_result(evaluation_id: int, db: Session = Depends(get_db)) -> Dict[str, Any]:
    """评估结果：状态、进度、四维度得分、聚合指标、逐条明细。"""
    evaluation = db.query(Evaluation).filter(Evaluation.id == evaluation_id).first()
    if not evaluation:
        raise HTTPException(status_code=404, detail="Evaluation not found")

    running = evaluation.status == "running"

    metrics: Dict[str, Any] = {}
    if not running:
        try:
            from app.monitoring.metrics import MetricsCollector

            metrics = MetricsCollector(db).collect_evaluation_metrics(evaluation_id)
        except Exception as exc:  # noqa: BLE001 - 指标聚合失败不应挡住主结果
            metrics = {"_error": str(exc)}

    executions = _collect_executions(db, evaluation_id)
    metrics["applied_dimensions"] = _applied_dimensions(evaluation)

    return {
        "id": evaluation.id,
        "agent_name": evaluation.agent_name,
        "status": evaluation.status,
        "running": running,
        "progress": {
            "total": evaluation.total_cases,
            "successful": evaluation.successful_cases,
            "failed": evaluation.failed_cases,
            "done": len(executions),
        },
        "scores": {
            "overall": evaluation.overall_score,
            "quality": evaluation.quality_score,
            "performance": evaluation.performance_score,
            "security": evaluation.security_score,
            "task_completion": evaluation.task_completion_score,
        },
        "totals": {
            "cost": evaluation.total_cost,
            "tokens": evaluation.total_tokens,
            "started_at": evaluation.started_at,
            "completed_at": evaluation.completed_at,
        },
        "metrics": metrics,
        "executions": executions,
    }


@router.get("/history")
def history(limit: int = Query(50, ge=1, le=200), db: Session = Depends(get_db)) -> List[Dict[str, Any]]:
    """历史评估列表，供前端挑选两次评估做对比。"""
    rows = (
        db.query(Evaluation)
        .order_by(Evaluation.id.desc())
        .limit(limit)
        .all()
    )
    result = []
    for evaluation in rows:
        item = _summary(evaluation)
        item["dataset_name"] = None
        if evaluation.dataset_id:
            dataset = db.query(Dataset).filter(Dataset.id == evaluation.dataset_id).first()
            if dataset:
                item["dataset_name"] = dataset.name
        result.append(item)
    return result


@router.get("/compare")
def compare(
    current: int = Query(..., description="当前评估 ID"),
    baseline: int = Query(..., description="基线评估 ID"),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """两次评估对比，delta = current - baseline。"""
    target = db.query(Evaluation).filter(Evaluation.id == current).first()
    base = db.query(Evaluation).filter(Evaluation.id == baseline).first()
    if not target or not base:
        raise HTTPException(status_code=404, detail="Evaluation not found")

    delta: Dict[str, Optional[float]] = {}
    for key, value in _summary(target)["scores"].items():
        base_value = _summary(base)["scores"].get(key)
        delta[key] = None if (value is None or base_value is None) else round(value - base_value, 2)

    return {"current": _summary(target), "baseline": _summary(base), "delta": delta}


_REPORT_CSS = """
body{margin:0;padding:32px 24px;background:#f7f7f5;color:#2c2c2a;
font-family:-apple-system,"Segoe UI","Microsoft YaHei",system-ui,sans-serif;font-size:14px;line-height:1.6}
.wrap{max-width:920px;margin:0 auto}
h1{font-size:20px;font-weight:500;margin:0 0 4px}
h2{font-size:15px;font-weight:500;margin:28px 0 10px}
.sub{color:#6b6a66;margin:0 0 20px;font-size:13px}
.card{background:#fff;border:1px solid #e3e2de;border-radius:12px;padding:18px 20px;margin-bottom:14px}
.overall{font-size:38px;font-weight:500;line-height:1.1}
.grid{display:flex;gap:12px;flex-wrap:wrap;margin-top:14px}
.score{flex:1 1 140px;background:#fafaf8;border-radius:8px;padding:12px 14px}
.score .k{font-size:12px;color:#6b6a66}
.score .v{font-size:22px;font-weight:500}
.kv{display:flex;gap:18px;flex-wrap:wrap;font-size:13px;color:#6b6a66;margin-top:10px}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{text-align:left;padding:8px 10px;border-bottom:1px solid #e3e2de;vertical-align:top}
th{color:#6b6a66;font-weight:400;font-size:12px}
.chain{display:flex;gap:6px;flex-wrap:wrap;margin:8px 0}
.chain span{background:#f1efe8;border-radius:6px;padding:2px 9px;font-size:12px}
.arrow{color:#a3a29e}
.reason{font-size:13px;color:#5f5e5a;background:#fafaf8;border:1px solid #e3e2de;
border-radius:8px;padding:10px;margin-top:8px;white-space:pre-wrap}
.tag{display:inline-block;padding:1px 8px;border-radius:20px;font-size:12px}
.ok{background:#e1f5ee;color:#0f6e56}
.bad{background:#fcebeb;color:#a32d2d}
"""


def _build_report_html(data: Dict[str, Any]) -> str:
    """生成可离线打开的单文件评测报告。"""
    def esc(value: Any) -> str:
        return html.escape(str(value if value is not None else ""))

    def fmt(value: Any) -> str:
        return "--" if value is None else f"{float(value):.1f}"

    def iso(value: Any) -> str:
        if not value:
            return "--"
        text = value if isinstance(value, str) else value.isoformat()
        return esc(text.replace("T", " ")[:19])

    scores = data["scores"]
    progress = data["progress"]
    parts: List[str] = []

    parts.append(
        '<div class="card"><h1>Agent 评测报告</h1>'
        f'<p class="sub">{esc(data["agent_name"])}　任务 #{data["id"]}　'
        f'{esc(data["status"])}　{iso(data.get("started_at"))}</p>'
        f'<div class="overall">{fmt(scores["overall"])}<small style="font-size:14px;color:#6b6a66"> / 100 综合分</small></div>'
        '<div class="kv">'
        f'<span>用例 {esc(progress["total"] or 0)} 条</span>'
        f'<span>成功 {esc(progress["successful"] or 0)}</span>'
        f'<span>失败 {esc(progress["failed"] or 0)}</span>'
        f'<span>Token {esc(data["totals"]["tokens"] or 0)}</span>'
        f'<span>成本 ${float(data["totals"]["cost"] or 0):.6f}</span>'
        "</div>"
        '<div class="grid">'
    )
    for label, key in (("质量", "quality"), ("性能", "performance"),
                       ("安全", "security"), ("任务完成度", "task_completion")):
        parts.append(
            f'<div class="score"><div class="k">{label}</div>'
            f'<div class="v">{fmt(scores[key])}</div></div>'
        )
    parts.append("</div></div>")

    # 哪些维度真正参与了总分：红队集只开安全维度，其余显示 -- 并说明原因，
    # 免得读数的人以为「质量 0 分」
    applied = (data.get("metrics") or {}).get("applied_dimensions")
    if isinstance(applied, list) and 0 < len(applied) < 4:
        dim_names = {
            "quality": "质量",
            "performance": "性能",
            "security": "安全",
            "task_completion": "任务完成度",
        }
        joined = "、".join(dim_names.get(x, x) for x in applied)
        parts.append(
            '<div class="card"><p style="margin:0;font-size:12px;color:#5f5e5a">'
            f"本次只有 <b>{esc(joined)}</b> 参与总分合成；"
            "其余维度对这批用例不适用（用例权重为 0），显示为 --。</p></div>"
        )

    if data.get("metrics"):
        parts.append(
            '<div class="card"><h2>聚合指标</h2><pre style="white-space:pre-wrap;margin:0;'
            'font-size:12px;color:#5f5e5a">'
            + esc(json.dumps(data["metrics"], ensure_ascii=False, indent=2))
            + "</pre></div>"
        )

    parts.append('<div class="card"><h2>逐条明细</h2><table>'
                 "<thead><tr><th>#</th><th>输入 / 结果</th><th>安全分</th><th>耗时</th></tr></thead><tbody>")
    for item in data.get("executions", []):
        tag = "ok" if item["status"] == "completed" else "bad"
        chain = "".join(
            f'<span>{esc(tool["name"])}</span><span class="arrow">→</span>'
            for tool in item.get("tools", [])
        )
        if chain:
            chain = chain[:-len('<span class="arrow">→</span>')]
        reasoning = item.get("reasoning") or {}
        sources = item.get("sources") or {}
        indicators = item.get("indicators") or {}
        blocks = ""
        source_note = {
            "estimated": "Token 与成本为本地估算值（被测服务未上报用量，估算不含其内部 RAG 上下文），仅供横向比较，勿当作真实账单",
            "missing": "未获得 Token 用量，性能维度已降级为仅按延迟计分",
        }.get(sources.get("token", ""))
        if source_note:
            blocks += f'<div class="reason"><b>数据口径</b>　{esc(source_note)}</div>'
        for label, key in (("质量", "quality"), ("安全", "security"), ("任务", "task")):
            if reasoning.get(key):
                blocks += f'<div class="reason"><b>{label}</b>　{esc(reasoning[key])}</div>'
        if item.get("error"):
            blocks += f'<div class="reason"><b>错误</b>　{esc(item["error"])}</div>'

        # 逐条安全分 + 命中的具体风险类型，让报告能回答「这条为什么扣分」
        sec_value = indicators.get("security_score")
        sec_cell = "--" if sec_value is None else f"{float(sec_value):.2f}"
        risk_flags = [
            label for key, label in (
                ("prompt_leakage", "提示词泄露"),
                ("dangerous_operation", "危险操作"),
                ("privilege_escalation", "权限越界"),
            ) if indicators.get(key)
        ]
        if risk_flags:
            sec_cell += (
                '<div style="font-size:11px;line-height:1.5;color:#a32d2d">'
                + esc("、".join(risk_flags)) + "</div>"
            )

        parts.append(
            "<tr><td>" + str(item["id"]) + "</td><td>"
            + '<div style="margin-bottom:4px">' + esc(item["input_prompt"]) + "</div>"
            + '<span class="tag ' + tag + '">' + esc(item["status"]) + "</span>"
            + ('<div class="chain">' + chain + "</div>" if chain else "")
            + '<div style="color:#5f5e5a">' + esc(item.get("response") or "") + "</div>"
            + blocks
            + "</td><td>" + sec_cell + "</td><td>"
            + (f'{float(item["duration"]):.2f}s' if item.get("duration") else "--")
            + "</td></tr>"
        )
    parts.append("</tbody></table></div>")

    parts.append(
        '<p class="sub" style="margin-top:20px">由 agent-eval 生成 · '
        f'{esc(datetime.utcnow().strftime("%Y-%m-%d %H:%M"))} UTC</p>'
    )

    return (
        '<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Agent 评测报告 #" + str(data["id"]) + "</title>"
        "<style>" + _REPORT_CSS + "</style></head><body><div class=\"wrap\">"
        + "".join(parts) + "</div></body></html>"
    )


@router.post("/cancel/{evaluation_id}")
def cancel(evaluation_id: int, db: Session = Depends(get_db)) -> Dict[str, Any]:
    """把卡在 running 的任务标记为 cancelled。

    后台任务跑在请求进程里，服务重启或进程被杀后状态会永远停在 running，
    这里给一个人工收口的方式（不会强杀仍在运行的线程）。
    """
    evaluation = db.query(Evaluation).filter(Evaluation.id == evaluation_id).first()
    if not evaluation:
        raise HTTPException(status_code=404, detail="Evaluation not found")

    if evaluation.status != "running":
        return {"id": evaluation.id, "status": evaluation.status, "changed": False}

    evaluation.status = "cancelled"
    evaluation.completed_at = datetime.utcnow()
    db.commit()
    return {"id": evaluation.id, "status": "cancelled", "changed": True}


@router.get("/report/{evaluation_id}", response_class=HTMLResponse)
def report(evaluation_id: int, db: Session = Depends(get_db)) -> HTMLResponse:
    """导出单文件 HTML 报告，可离线打开或直接发给别人。"""
    evaluation = db.query(Evaluation).filter(Evaluation.id == evaluation_id).first()
    if not evaluation:
        raise HTTPException(status_code=404, detail="Evaluation not found")

    metrics: Dict[str, Any] = {}
    if evaluation.status != "running":
        try:
            from app.monitoring.metrics import MetricsCollector

            metrics = MetricsCollector(db).collect_evaluation_metrics(evaluation_id)
        except Exception:  # noqa: BLE001
            metrics = {}

    payload = _summary(evaluation)
    payload["metrics"] = metrics
    metrics["applied_dimensions"] = _applied_dimensions(evaluation)
    payload["executions"] = _collect_executions(db, evaluation_id)

    return HTMLResponse(content=_build_report_html(payload))


# ---------------------------------------------------------------- 多次跑一致率
# 单次跑分的方差主要来自被测 Agent 自身（同一问题不同轮次行为可能完全不同），
# 只看总分会把这种漂移掩盖掉：一条误报扣的分，正好被另一条漏报补回来。
# 所以这里按「逐条」比对多轮结果，给出判定一致率与回答一致率。

_FLAG_LABELS = (
    ("prompt_leakage", "泄露"),
    ("dangerous_operation", "危险操作"),
    ("privilege_escalation", "越权"),
)


def _case_snapshots(db: Session, evaluation_id: int) -> Dict[int, Dict[str, Any]]:
    """某一轮里每条用例的判定快照（按 test_case_id 索引）。"""
    rows = (
        db.query(AgentExecution, TestCase.input_prompt, EvaluationMetric)
        .outerjoin(TestCase, TestCase.id == AgentExecution.test_case_id)
        .outerjoin(EvaluationMetric, EvaluationMetric.execution_id == AgentExecution.id)
        .filter(AgentExecution.evaluation_id == evaluation_id)
        .all()
    )
    snapshots: Dict[int, Dict[str, Any]] = {}
    for execution, input_prompt, metric in rows:
        flags = [
            label
            for key, label in _FLAG_LABELS
            if metric is not None and getattr(metric, key, False)
        ]
        text = (execution.final_response or "").strip()
        tools = [
            c.get("name")
            for c in (execution.tool_calls or [])
            if isinstance(c, dict) and c.get("name")
        ]
        snapshots[execution.test_case_id] = {
            "prompt": input_prompt or "",
            "status": execution.status,
            "security_score": getattr(metric, "security_score", None),
            "accuracy": getattr(metric, "accuracy", None),
            "completeness": getattr(metric, "completeness", None),
            "relevance": getattr(metric, "relevance", None),
            "flags": flags,
            "digest": hashlib.md5(text.encode("utf-8")).hexdigest()[:8],
            "tool_names": tools,
        }
    return snapshots


def _spread(values: List[Optional[float]]) -> Optional[float]:
    """一组分数的极差（max-min），全为空时返回 None。"""
    nums = [float(v) for v in values if v is not None]
    if not nums:
        return None
    return round(max(nums) - min(nums), 4)


def _score_stats(values: List[Optional[float]]) -> Dict[str, Any]:
    nums = [float(v) for v in values if v is not None]
    if not nums:
        return {"mean": None, "min": None, "max": None, "spread": None, "stdev": None}
    mean = sum(nums) / len(nums)
    var = sum((x - mean) ** 2 for x in nums) / len(nums)
    return {
        "mean": round(mean, 2),
        "min": round(min(nums), 2),
        "max": round(max(nums), 2),
        "spread": round(max(nums) - min(nums), 2),
        "stdev": round(var ** 0.5, 2),
    }


def _build_consistency(ids: List[int], db: Session) -> Dict[str, Any]:
    """把多轮评估按用例对齐，算判定一致率 / 行为一致率。"""
    evaluations = (
        db.query(Evaluation).filter(Evaluation.id.in_(ids)).order_by(Evaluation.id).all()
    )
    if len(evaluations) != len(ids):
        found = {e.id for e in evaluations}
        missing = [str(i) for i in ids if i not in found]
        raise HTTPException(status_code=404, detail=f"评估不存在: {','.join(missing)}")

    dataset_ids = {e.dataset_id for e in evaluations}
    if len(dataset_ids) > 1:
        raise HTTPException(
            status_code=400,
            detail=f"只能对比同一数据集的多轮结果，当前包含 dataset {sorted(dataset_ids)}",
        )

    runs = [
        {
            "id": e.id,
            "agent_name": e.agent_name,
            "status": e.status,
            "overall": e.overall_score,
            "security": e.security_score,
            "completed_at": e.completed_at.isoformat() if e.completed_at else None,
        }
        for e in evaluations
    ]

    snapshots = [_case_snapshots(db, e.id) for e in evaluations]
    case_ids: List[int] = []
    for snap in snapshots:
        for case_id in snap:
            if case_id not in case_ids:
                case_ids.append(case_id)

    cases: List[Dict[str, Any]] = []
    for case_id in case_ids:
        per_run = [snap.get(case_id) for snap in snapshots]
        prompts = [r["prompt"] for r in per_run if r]
        flag_sets = [tuple(sorted(r["flags"])) for r in per_run if r]
        digests = [r["digest"] for r in per_run if r]
        tools = [tuple(r["tool_names"]) for r in per_run if r]
        cases.append(
            {
                "test_case_id": case_id,
                "prompt": prompts[0] if prompts else "",
                "runs": [
                    None if r is None else {
                        "security_score": r["security_score"],
                        "accuracy": r["accuracy"],
                        "flags": r["flags"],
                        "digest": r["digest"],
                        "tool_names": r["tool_names"],
                    }
                    for r in per_run
                ],
                # 三层一致性，粒度从粗到细
                # 1) 判定一致：每轮「有没有安全问题」的结论一样
                "judge_consistent": len(set(flag_sets)) <= 1,
                # 分数稳定：每轮准确率极差 ≤ 0.1（10 个百分点）。
                # 功能集是连续打分，看波动幅度比看"有没有标记"有意义
                "accuracy_spread": _spread(
                    [r.get("accuracy") for r in per_run if r]
                ),
                "score_stable": (_spread([r.get("accuracy") for r in per_run if r]) or 0.0) <= 0.1,
                # 2) 工具链一致：调用的工具及其顺序一样（Agent 走了同一条路径）
                "tool_consistent": len(set(tools)) <= 1,
                # 3) 文本一致：回答逐字相同（temperature>0 时基本不可能）
                "text_identical": len(set(digests)) <= 1,
            }
        )

    total = len(cases)
    judge_ok = sum(1 for c in cases if c["judge_consistent"])
    stable_ok = sum(1 for c in cases if c.get("score_stable"))
    tool_ok = sum(1 for c in cases if c["tool_consistent"])
    text_ok = sum(1 for c in cases if c["text_identical"])
    flagged = [c for c in cases if any((r or {}).get("flags") for r in c["runs"])]

    return {
        "evaluation_ids": [e.id for e in evaluations],
        "dataset_id": dataset_ids.pop(),
        "runs": runs,
        "security_stats": _score_stats([e.security_score for e in evaluations]),
        "overall_stats": _score_stats([e.overall_score for e in evaluations]),
        "summary": {
            "total_cases": total,
            "judge_consistent": judge_ok,
            "judge_rate": round(judge_ok / total, 4) if total else None,
            "score_stable": stable_ok,
            "score_stable_rate": round(stable_ok / total, 4) if total else None,
            "tool_consistent": tool_ok,
            "tool_rate": round(tool_ok / total, 4) if total else None,
            "text_identical": text_ok,
            "text_rate": round(text_ok / total, 4) if total else None,
            "flagged_cases": len(flagged),
        },
        "cases": cases,
    }


@router.get("/consistency")
def consistency(ids: str = Query(..., description="逗号分隔的评估 id，需同一数据集"),
                db: Session = Depends(get_db)) -> Dict[str, Any]:
    """多次跑的逐条一致率：判定是否稳定、回答是否相同。"""
    try:
        id_list = [int(x) for x in ids.replace(" ", "").split(",") if x]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="ids 必须是逗号分隔的整数") from exc
    if len(id_list) < 2:
        raise HTTPException(status_code=400, detail="至少需要 2 个评估才能比较一致率")
    return _build_consistency(id_list, db)


@router.get("/consistency/report", response_class=HTMLResponse)
def consistency_report(ids: str = Query(..., description="逗号分隔的评估 id"),
                       db: Session = Depends(get_db)) -> HTMLResponse:
    """多轮一致率 HTML 报告：总分对比 + 逐条漂移明细。

    路径故意放在 /consistency 下，而不是 /report/consistency ——
    /report/{evaluation_id} 已把 evaluation_id 声明为 int，先注册会把它吃掉，
    请求直接 422（int_parsing）。
    """
    id_list = [int(x) for x in ids.replace(" ", "").split(",") if x]
    data = _build_consistency(id_list, db)
    return HTMLResponse(content=_build_consistency_html(data))


def _build_consistency_html(data: Dict[str, Any]) -> str:
    esc = lambda v: html.escape(str(v if v is not None else ""))  # noqa: E731
    fmt = lambda v: "--" if v is None else f"{float(v):.1f}"  # noqa: E731

    runs = data["runs"]
    sec = data["security_stats"]
    parts: List[str] = [
        '<div class="card"><h1>多轮一致率报告</h1>'
        f'<p class="sub">数据集 #{data["dataset_id"]} · 共 {len(runs)} 轮 · '
        f'{data["summary"]["total_cases"]} 条用例</p></div>'
    ]

    parts.append('<div class="card"><h2>各轮安全分</h2><table><thead><tr>'
                 '<th>轮次</th><th>评估 ID</th><th>安全分</th><th>完成时间</th>'
                 "</tr></thead><tbody>")
    for idx, run in enumerate(runs, 1):
        parts.append(
            f"<tr><td>第 {idx} 轮</td><td>#{run['id']}</td>"
            f"<td>{fmt(run['security'])}</td>"
            f"<td>{esc((run['completed_at'] or '')[:19])}</td></tr>"
        )
    parts.append("</tbody></table>")
    parts.append(
        '<div class="kv" style="margin-top:12px">'
        f'<span>均值 {fmt(sec["mean"])}</span>'
        f'<span>极差 {fmt(sec["spread"])}</span>'
        f'<span>标准差 {fmt(sec["stdev"])}</span>'
        f'<span>范围 {fmt(sec["min"])} ~ {fmt(sec["max"])}</span>'
        "</div></div>"
    )

    summary = data["summary"]
    parts.append(
        '<div class="card"><h2>一致率</h2><div class="kv">'
        f'<span>判定一致 {summary["judge_consistent"]}/{summary["total_cases"]}'
        f'（{summary["judge_rate"] * 100:.1f}%）</span>'
        f'<span>工具链一致 {summary["tool_consistent"]}/{summary["total_cases"]}'
        f'（{summary["tool_rate"] * 100:.1f}%）</span>'
        f'<span>回答逐字相同 {summary["text_identical"]}/{summary["total_cases"]}'
        f'（{summary["text_rate"] * 100:.1f}%）</span>'
        f'<span>分数稳定（准确率极差≤0.1） {summary["score_stable"]}/{summary["total_cases"]}'
        f'（{summary["score_stable_rate"] * 100:.1f}%）</span>'
        f'<span>曾被判定有问题的用例 {summary["flagged_cases"]} 条</span>'
        "</div>"
        '<p class="sub" style="margin:10px 0 0">'
        "判定一致 = 每轮「有没有安全问题」结论相同（最关键）；"
        "工具链一致 = 调用的工具及顺序相同；"
        "回答逐字相同 = 文本完全相同（temperature&gt;0 时检出率天然很低，"
        "该项为 0 不代表 Agent 行为异常）。</p></div>"
    )

    header = "".join(
        f'<th>第 {i} 轮</th>' for i in range(1, len(runs) + 1)
    )
    parts.append('<div class="card"><h2>逐条明细</h2><table><thead><tr>'
                 f"<th>#</th><th>用例</th>{header}<th>判定</th><th>准确率极差</th>"
                 "<th>工具链</th><th>文本</th>"
                 "</tr></thead><tbody>")
    for idx, case in enumerate(data["cases"], 1):
        cells = ""
        for run in case["runs"]:
            if run is None:
                cells += "<td>--</td>"
                continue
            flag_text = "、".join(run["flags"]) if run["flags"] else "—"
            score = fmt(run["security_score"])
            color = "#a32d2d" if run["flags"] else "#085041"
            cells += (
                f"<td>{score}<div style=\"font-size:11px;color:{color}\">"
                f"{esc(flag_text)}</div></td>"
            )
        judge = '<span class="tag ok">一致</span>' if case["judge_consistent"] else \
            '<span class="tag bad">漂移</span>'
        spread_val = case.get("accuracy_spread")
        spread_cell = "--" if spread_val is None else f"{spread_val:.2f}"
        if spread_val is not None and spread_val > 0.1:
            spread_cell = f'<span style="color:#a32d2d">{spread_cell}</span>'
        tools_tag = '<span class="tag ok">一致</span>' if case["tool_consistent"] else \
            '<span class="tag bad">不同</span>'
        text_tag = '<span class="tag ok">相同</span>' if case["text_identical"] else \
            '<span class="tag">不同</span>'
        parts.append(
            f"<tr><td>{idx}</td><td>{esc(case['prompt'][:46])}</td>{cells}"
            f"<td>{judge}</td><td>{spread_cell}</td><td>{tools_tag}</td><td>{text_tag}</td></tr>"
        )
    parts.append("</tbody></table></div>")

    parts.append(
        '<div class="card"><h2>怎么读这份报告</h2>'
        '<p class="sub" style="margin:0">安全分均值接近，不代表结论稳定。'
        "一份「总分很稳、逐条大量漂移」的结果，说明分数是靠误报与漏报互相抵消凑出来的。"
        "对外汇报时应当同时给出判定一致率；低于 80% 说明要么被测 Agent 行为不确定，"
        "要么判定口径不稳定，两者都需要先查清再下结论。</p></div>"
    )

    parts.append(
        '<p class="sub" style="margin-top:20px">由 agent-eval 生成 · '
        f'{html.escape(datetime.utcnow().strftime("%Y-%m-%d %H:%M"))} UTC</p>'
    )

    return (
        '<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>多轮一致率报告</title>"
        "<style>" + _REPORT_CSS + "</style></head><body><div class=\"wrap\">"
        + "".join(parts) + "</div></body></html>"
    )
