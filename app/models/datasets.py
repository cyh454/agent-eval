from sqlalchemy import Column, Integer, String, Text, JSON, DateTime, Float, ForeignKey
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
from datetime import datetime
from app.database import Base
from app.core.config import get_settings

# 与 .env 的 EMBEDDING_DIMENSION 保持一致，避免各处硬编码维度导致落地失败
_EMBEDDING_DIMENSION = get_settings().EMBEDDING_DIMENSION

class Dataset(Base):
    __tablename__ = "datasets"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, index=True)
    description = Column(Text)
    category = Column(String(50))  # 如: qa, coding, reasoning, multi_tool
    
    # 元数据
    config = Column(JSON, default={})  # 数据集特定配置
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # 关系
    test_cases = relationship("TestCase", back_populates="dataset", cascade="all, delete-orphan")
    evaluations = relationship("Evaluation", back_populates="dataset", cascade="all, delete-orphan")

class TestCase(Base):
    __tablename__ = "test_cases"
    
    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)
    
    # 输入
    input_prompt = Column(Text, nullable=False)
    input_embedding = Column(Vector(_EMBEDDING_DIMENSION))
    
    # 期望输出
    ground_truth = Column(Text, nullable=False)
    ground_truth_embedding = Column(Vector(_EMBEDDING_DIMENSION))
    expected_tools = Column(JSON, default=[])  # 期望调用的工具列表
    expected_steps = Column(JSON, default=[])  # 期望的执行步骤
    
    # 评估权重
    quality_weight = Column(Float, default=1.0)
    performance_weight = Column(Float, default=1.0)
    security_weight = Column(Float, default=1.0)
    task_weight = Column(Float, default=1.0)
    
    # 安全规则
    security_rules = Column(JSON, default={})
    
    # 元数据
    difficulty = Column(String(20))  # easy, medium, hard
    tags = Column(JSON, default=[])
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # 关系
    dataset = relationship("Dataset", back_populates="test_cases")
    # 必须级联删除执行记录，否则删除数据集时 ORM 会尝试把非空外键置为 NULL 而报错
    executions = relationship(
        "AgentExecution", back_populates="test_case", cascade="all, delete-orphan"
    )
