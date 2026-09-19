from sqlalchemy import (
    Boolean,
    Column,
    Integer,
    String,
    Text,
    JSON,
    DateTime,
    Float,
    ForeignKey,
)
from sqlalchemy.orm import relationship
from datetime import datetime
from app.database import Base

class Evaluation(Base):
    __tablename__ = "evaluations"
    
    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)
    
    # 执行信息
    agent_name = Column(String(100))
    agent_version = Column(String(50))
    agent_config = Column(JSON, default={})
    
    # 时间信息
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    
    # 统计信息
    total_cases = Column(Integer, default=0)
    successful_cases = Column(Integer, default=0)
    failed_cases = Column(Integer, default=0)
    
    # 综合得分 (0-100)
    overall_score = Column(Float)
    
    # 各维度得分 (0-100)
    quality_score = Column(Float)
    performance_score = Column(Float)
    security_score = Column(Float)
    task_completion_score = Column(Float)
    
    # 详细指标
    metrics = Column(JSON, default={})  # 包含所有详细指标
    
    # 成本统计
    total_cost = Column(Float, default=0.0)
    total_tokens = Column(Integer, default=0)
    
    # 状态
    status = Column(String(20))  # running, completed, partial, failed
    
    # 关系
    dataset = relationship("Dataset", back_populates="evaluations")
    # 级联删除，保证 DELETE /datasets/{id} 能清空执行记录与逐条指标
    executions = relationship(
        "AgentExecution", back_populates="evaluation", cascade="all, delete-orphan"
    )
    # 注意：类名 metrics 已被 JSON 列占用，关系只能用 metric_records
    metric_records = relationship(
        "EvaluationMetric", back_populates="evaluation", cascade="all, delete-orphan"
    )
    
class EvaluationMetric(Base):
    __tablename__ = "evaluation_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    evaluation_id = Column(Integer, ForeignKey("evaluations.id"), nullable=False)
    execution_id = Column(Integer, ForeignKey("agent_executions.id"), nullable=False)
    
    # 质量指标
    accuracy = Column(Float)           # 准确性 (0-1)
    completeness = Column(Float)       # 完整性 (0-1)
    hallucination_rate = Column(Float) # 幻觉率 (0-1)
    relevance = Column(Float)          # 相关性/是否跑题 (0-1)
    
    # 性能指标
    latency = Column(Float)            # 响应延迟(秒)
    throughput = Column(Float)         # 吞吐量(requests/min)
    token_efficiency = Column(Float)   # Token效率
    cost = Column(Float)               # 成本
    
    # 安全指标
    prompt_leakage = Column(Boolean, default=False)    # Prompt泄露
    dangerous_operation = Column(Boolean, default=False) # 危险操作
    privilege_escalation = Column(Boolean, default=False) # 权限越界
    
    # 任务完成度
    task_success = Column(Boolean)     # 任务是否成功
    steps_correctness = Column(Float)  # 步骤正确性
    tool_usage_correctness = Column(Float) # 工具使用正确性
    
    # 评估详情
    quality_reasoning = Column(Text)   # 质量评估理由
    security_reasoning = Column(Text)  # 安全评估理由
    task_reasoning = Column(Text)      # 任务评估理由
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # 关系
    evaluation = relationship("Evaluation", back_populates="metric_records")
