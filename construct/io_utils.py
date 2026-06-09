from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from .models import CaseRecord, KnowledgeNode
from .utils import ensure_directory


def load_cases(path: Path) -> list[CaseRecord]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases: list[CaseRecord] = []
    for case_id, item in payload.items():
        cases.append(
            CaseRecord(
                case_id=case_id,
                case_name=item["case_name"],
                text=item["text"],
            )
        )
    return cases


def load_seed_l1(path: Path) -> list[KnowledgeNode]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    nodes: list[KnowledgeNode] = []
    for item in payload:
        nodes.append(
            KnowledgeNode(
                name=item["name"],
                trigger=item["trigger"],
                background=item["background"],
            )
        )
    return nodes


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    ensure_directory(path.parent)
    path.write_text(
        json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_text(path: Path, text: str) -> None:
    ensure_directory(path.parent)
    path.write_text(text, encoding="utf-8")


def write_tree_outputs(
    root: KnowledgeNode,
    tree_path: Path,
    debug_tree_path: Path,
) -> None:
    write_json(tree_path, root.to_tree_dict())
    write_json(debug_tree_path, root.to_debug_dict())
