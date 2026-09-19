"""ORM 模型集合。

目录已由 ``app/model`` 统一更名为 ``app/models``，以便与代码中的
``from app.models import ...`` 引用保持一致（原有 9 处引用全部指向 models）。
导入本包会注册全部映射，供 ``Base.metadata.create_all`` 使用。
"""

from app.models.datasets import Dataset, TestCase
from app.models.evaluation import Evaluation, EvaluationMetric
from app.models.execution import AgentExecution
from app.models.security import SecurityLog, SecurityRule

__all__ = [
    "Dataset",
    "TestCase",
    "Evaluation",
    "EvaluationMetric",
    "AgentExecution",
    "SecurityLog",
    "SecurityRule",
]
