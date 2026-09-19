from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import relationship
from datetime import datetime
from app.database import Base

class SecurityLog(Base):
    __tablename__ = "security_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    execution_id = Column(Integer, ForeignKey("agent_executions.id"), nullable=False)
    
    # 安全事件
    event_type = Column(String(50))  # prompt_leakage, dangerous_op, privilege_escalation, other
    severity = Column(String(20))    # low, medium, high, critical
    
    # 事件详情
    description = Column(Text)
    detected_at = Column(DateTime, default=datetime.utcnow)
    detection_method = Column(String(50))  # rule, llm, hybrid
    
    # 上下文
    context = Column(JSON, default={})
    triggered_by = Column(String(100))  # 触发规则名称
    
    # 处理状态
    status = Column(String(20))  # detected, investigating, resolved, ignored
    
    # 关系
    execution = relationship("AgentExecution", back_populates="security_logs")

class SecurityRule(Base):
    __tablename__ = "security_rules"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True)
    category = Column(String(50))  # prompt_leakage, dangerous_op, privilege_escalation
    
    # 规则定义
    rule_type = Column(String(20))  # keyword, pattern, semantic, custom
    pattern = Column(Text)  # 正则表达式或关键词
    threshold = Column(Float)  # 阈值(用于语义检测)
    
    # 配置
    enabled = Column(Boolean, default=True)
    severity = Column(String(20))
    
    # 元数据
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
