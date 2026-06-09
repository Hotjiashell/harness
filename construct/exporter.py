from __future__ import annotations

from pathlib import Path

from .io_utils import write_json
from .models import CaseRecord, KnowledgeNode
from .utils import ensure_directory, slugify


def export_skill_tree(
    root: KnowledgeNode,
    cases_by_id: dict[str, CaseRecord],
    target_dir: Path,
) -> None:
    ensure_directory(target_dir)
    root_dir = target_dir / "root"
    _export_node(root, cases_by_id, root_dir, sibling_index=0)


def _export_node(
    node: KnowledgeNode,
    cases_by_id: dict[str, CaseRecord],
    directory: Path,
    sibling_index: int,
) -> None:
    ensure_directory(directory)
    node_payload = {
        "name": node.name,
        "trigger": node.trigger,
        "background": node.background,
        "depth": node.depth,
        "path": node.path,
        "child_count": len(node.children),
    }
    write_json(directory / "node.json", node_payload)
    write_json(
        directory / "cases.json",
        [
            cases_by_id[case_id].to_dict()
            for case_id in node.case_ids
            if case_id in cases_by_id
        ],
    )

    for index, child in enumerate(node.children, start=1):
        fallback = f"node-{index}"
        child_slug = slugify(child.name, fallback=fallback)
        child_dir = directory / f"{index:02d}_{child_slug}"
        _export_node(child, cases_by_id, child_dir, sibling_index=index)
