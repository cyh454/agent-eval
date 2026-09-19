from abc import ABC, abstractmethod
from typing import Dict, Any
from app.models.execution import AgentExecution
from app.models.datasets import TestCase

class BaseEvaluator(ABC):
    """评估器基类"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
    
    @abstractmethod
    def evaluate(self, execution: AgentExecution, test_case: TestCase) -> Dict[str, Any]:
        """
        执行评估
        
        Returns:
            Dict containing:
                - score: float (0-1)
                - metrics: Dict[str, Any]
                - reasoning: str
        """
        pass
    
    @abstractmethod
    def get_dimension_name(self) -> str:
        """返回评估维度名称"""
        pass
