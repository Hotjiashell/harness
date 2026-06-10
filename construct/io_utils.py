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


def load_knowledge_node(
    path: Path,
    require_case_ids: bool = False,
) -> KnowledgeNode:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(payload).__name__}")
    return knowledge_node_from_dict(payload, require_case_ids=require_case_ids)


def knowledge_node_from_dict(
    payload: dict[str, Any],
    require_case_ids: bool = False,
    fallback_depth: int = 0,
    fallback_path: list[str] | None = None,
) -> KnowledgeNode:
    name = str(payload.get("name", "")).strip()
    if not name:
        raise ValueError("Knowledge node payload is missing a non-empty 'name'")

    raw_path = payload.get("path")
    if isinstance(raw_path, list) and raw_path:
        node_path = [str(part) for part in raw_path]
    else:
        node_path = [*(fallback_path or []), name]

    raw_depth = payload.get("depth")
    if isinstance(raw_depth, int):
        depth = raw_depth
    else:
        depth = fallback_depth

    raw_case_ids = payload.get("case_ids")
    if raw_case_ids is None:
        if require_case_ids:
            raise ValueError(
                f"Knowledge node '{name}' is missing required field 'case_ids'. "
                "Please use a debug tree file such as intermediate/05_initial_root.json "
                "or knowledge_tree_debug.json."
            )
        case_ids: list[str] = []
    elif isinstance(raw_case_ids, list):
        case_ids = [str(case_id) for case_id in raw_case_ids]
    else:
        raise ValueError(f"Knowledge node '{name}' has invalid 'case_ids': expected list")

    children_payload = payload.get("children") or []
    if not isinstance(children_payload, list):
        raise ValueError(f"Knowledge node '{name}' has invalid 'children': expected list")

    children = [
        knowledge_node_from_dict(
            child,
            require_case_ids=require_case_ids,
            fallback_depth=depth + 1,
            fallback_path=node_path,
        )
        for child in children_payload
        if isinstance(child, dict)
    ]

    return KnowledgeNode(
        name=name,
        trigger=str(payload.get("trigger", "")).strip(),
        background=str(payload.get("background", "")).strip(),
        children=children,
        case_ids=case_ids,
        depth=depth,
        path=node_path,
    )


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
