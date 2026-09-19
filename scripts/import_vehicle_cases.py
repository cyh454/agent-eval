"""把 vehicle-diagnosis-agent 的 data/eval_cases.json 导入成 agent-eval 数据集。

字段映射::

    message      -> input_prompt
    expect_any   -> ground_truth（拼成「参考答案要点：A、B、C」，该列非空必填）
    expected_tools / expected_steps / difficulty / tags  按需透传

用例文件默认取仓库内 ``data/eval_cases.json``（内置的车载诊断 30 条功能用例），
也可用 ``--cases`` 指向任意同格式的 JSON。

用法::

    python scripts/import_vehicle_cases.py
    python scripts/import_vehicle_cases.py --replace
    python scripts/import_vehicle_cases.py --cases data/security_cases.json \
        --name vehicle-diagnosis-security-redteam --category security_redteam

导入后记下返回的 dataset_id，创建评估任务时要用。
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from sqlalchemy.orm import Session  # noqa: E402

from app.database import SessionLocal  # noqa: E402
from app.models.datasets import Dataset, TestCase  # noqa: E402

DEFAULT_CASES = ROOT / "data" / "eval_cases.json"
ALLOWED_DIFFICULTY = {"easy", "medium", "hard"}


def load_cases(path: Path) -> list:
    if not path.exists():
        print(f"用例文件不存在: {path}")
        print("用 --cases 指定其他路径，例如 --cases data/security_cases.json")
        sys.exit(1)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        print("用例文件必须是一个 JSON 数组")
        sys.exit(1)
    return data


def convert(raw: dict, extra_tags: Optional[List[str]] = None) -> dict:
    """把一条 vda 用例转成 TestCase 字段。"""
    expect = raw.get("expect_any") or []
    if expect:
        ground_truth = "参考答案要点：" + "、".join(str(k) for k in expect)
    else:
        ground_truth = "参考答案要点：（未指定关键词，请按领域常识评判）"

    difficulty = raw.get("difficulty")
    if difficulty is not None:
        difficulty = str(difficulty).strip().lower()
        if difficulty not in ALLOWED_DIFFICULTY:
            difficulty = None

    tags = list(extra_tags if extra_tags else ["vehicle-diagnosis"])
    tags.extend(str(t) for t in (raw.get("tags") or []))

    return {
        "input_prompt": str(raw.get("message") or "").strip(),
        "ground_truth": ground_truth,
        "expected_tools": list(raw.get("expected_tools") or []),
        "expected_steps": list(raw.get("expected_steps") or []),
        "difficulty": difficulty,
        "tags": tags,
    }


def import_dataset(db: Session, name: str, description: str, category: str,
                   cases: list, replace: bool, extra_tags: Optional[List[str]] = None) -> int:
    existing = db.query(Dataset).filter(Dataset.name == name).first()
    if existing:
        if not replace:
            print(f"数据集 '{name}' 已存在（id={existing.id}）。加 --replace 覆盖，或换 --name。")
            sys.exit(1)
        db.delete(existing)
        db.commit()
        print(f"已删除旧数据集 '{name}'（id={existing.id}）")

    dataset = Dataset(name=name, description=description, category=category, config={})
    db.add(dataset)
    db.flush()

    skipped = 0
    for raw in cases:
        payload = convert(raw, extra_tags)
        if not payload["input_prompt"]:
            skipped += 1
            continue
        db.add(TestCase(dataset_id=dataset.id, **payload))

    db.commit()
    db.refresh(dataset)

    count = db.query(TestCase).filter(TestCase.dataset_id == dataset.id).count()
    print(f"数据集创建成功: id={dataset.id} name='{name}' 用例数={count}"
          + (f"（跳过 {skipped} 条空输入）" if skipped else ""))
    return dataset.id


def main() -> None:
    parser = argparse.ArgumentParser(description="导入 vehicle-diagnosis-agent 评测用例")
    parser.add_argument("--cases", default=str(DEFAULT_CASES),
                        help="用例 JSON 路径（默认 data/eval_cases.json）")
    parser.add_argument("--name", default="vehicle-diagnosis-agent", help="数据集名称")
    parser.add_argument("--description", default="车辆故障诊断 Agent 评测集（来自 eval_cases.json）")
    parser.add_argument("--category", default="vehicle_diagnosis")
    parser.add_argument("--replace", action="store_true", help="同名数据集存在时先删除再导入")
    parser.add_argument("--tag", action="append",
                        help="附加标签，可重复；默认 vehicle-diagnosis")
    args = parser.parse_args()

    cases = load_cases(Path(args.cases))
    print(f"读取用例 {len(cases)} 条：{args.cases}")

    extra_tags = args.tag if args.tag else None
    db = SessionLocal()
    try:
        import_dataset(db, args.name, args.description, args.category,
                       cases, args.replace, extra_tags)
    finally:
        db.close()


if __name__ == "__main__":
    main()
