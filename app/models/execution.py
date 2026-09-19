from sqlalchemy import Column, Integer, String, Text, JSON, DateTime, Float, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime
from app.database import Base

class AgentExecution(Base):
    __tablename__ = "agent_executions"
    
    id = Column(Integer, primary_key=True, index=True)
    test_case_id = Column(Integer, ForeignKey("test_cases.id"), nullable=False)
    evaluation_id = Column(Integer, ForeignKey("evaluations.id"))
    
    # 执行信息
    agent_name = Column(String(100))
    agent_version = Column(String(50))
    
    # 时间指标
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    total_duration = Column(Float)  # 总耗时(秒)
    
    # Token 消耗
    total_tokens = Column(Integer, default=0)
    prompt_tokens = Column(Integer, default=0)
    completion_tokens = Column(Integer, default=0)
    estimated_cost = Column(Float, default=0.0)
    
    # 执行状态
    status = Column(String(20))  # running, completed, failed, timeout
    error_message = Column(Text)
    
    # 执行轨迹
    steps = Column(JSON, default=[])  # 每一步的详细记录
    tool_calls = Column(JSON, default=[])  # 工具调用记录
    intermediate_outputs = Column(JSON, default=[])  # 中间输出
    
    # 最终输出
    final_response = Column(Text)
    
    # 元数据
    # 列名保持 "metadata"，但 ORM 属性名必须避开 DeclarativeBase 保留字 metadata
    execution_metadata = Column("metadata", JSON, default={})
    
    # 关系
    test_case = relationship("TestCase", back_populates="executions")
    evaluation = relationship("Evaluation", back_populates="executions")
    security_logs = relationship("SecurityLog", back_populates="execution", cascade="all, delete-orphan")
