"""数据集与测试用例管理接口。"""

from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.datasets import Dataset, TestCase
from app.schemas.dataset import (
    DatasetCreate,
    DatasetDetailResponse,
    DatasetResponse,
    TestCaseCreate,
    TestCaseResponse,
)

router = APIRouter(prefix="/datasets", tags=["datasets"])


def _serialize_dataset(dataset: Dataset) -> DatasetResponse:
    payload = DatasetResponse.model_validate(dataset)
    payload.test_case_count = len(dataset.test_cases or [])
    return payload


@router.post("/", response_model=DatasetResponse, status_code=201)
def create_dataset(request: DatasetCreate, db: Session = Depends(get_db)):
    """创建数据集，可一次性带上测试用例。"""
    exists = db.query(Dataset).filter(Dataset.name == request.name).first()
    if exists:
        raise HTTPException(status_code=409, detail=f"Dataset '{request.name}' already exists")

    dataset = Dataset(
        name=request.name,
        description=request.description,
        category=request.category,
        config=request.config or {},
    )
    db.add(dataset)
    db.flush()

    for case in request.test_cases:
        db.add(TestCase(dataset_id=dataset.id, **case.model_dump()))

    db.commit()
    db.refresh(dataset)
    return _serialize_dataset(dataset)


@router.get("/", response_model=List[DatasetResponse])
def list_datasets(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    """数据集列表。"""
    datasets = db.query(Dataset).order_by(Dataset.id.desc()).offset(skip).limit(limit).all()
    return [_serialize_dataset(d) for d in datasets]


@router.get("/{dataset_id}", response_model=DatasetDetailResponse)
def get_dataset(dataset_id: int, db: Session = Depends(get_db)):
    """数据集详情（含全部测试用例）。"""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    detail = DatasetDetailResponse.model_validate(dataset)
    detail.test_case_count = len(dataset.test_cases or [])
    detail.test_cases = [TestCaseResponse.model_validate(tc) for tc in dataset.test_cases]
    return detail


@router.post("/{dataset_id}/test-cases", response_model=TestCaseResponse, status_code=201)
def add_test_case(
    dataset_id: int, request: TestCaseCreate, db: Session = Depends(get_db)
):
    """向已有数据集追加测试用例。"""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    test_case = TestCase(dataset_id=dataset_id, **request.model_dump())
    db.add(test_case)
    db.commit()
    db.refresh(test_case)
    return TestCaseResponse.model_validate(test_case)


@router.delete("/{dataset_id}", status_code=204)
def delete_dataset(dataset_id: int, db: Session = Depends(get_db)):
    """删除数据集（级联删除其测试用例与关联评估）。"""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    db.delete(dataset)
    db.commit()
    return Response(status_code=204)
