from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CaseRecord:
    case_id: str
    case_name: str
    text: str

    @property
    def content(self) -> str:
        return f"{self.case_name}\n{self.text}".strip()

    def to_dict(self) -> dict[str, str]:
        return {
            "case_id": self.case_id,
            "case_name": self.case_name,
            "text": self.text,
        }


@dataclass
class ClassificationResult:
    case_id: str
    belongs: bool
    category_name: str | None
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "belongs": self.belongs,
            "category_name": self.category_name,
            "reason": self.reason,
        }


@dataclass
class DiscoveryResult:
    case_id: str
    software_name: str
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "software_name": self.software_name,
            "description": self.description,
        }


@dataclass
class ClusterItem:
    case_id: str
    text: str
    source: str
    software_name: str = ""
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "text": self.text,
            "source": self.source,
            "software_name": self.software_name,
            "description": self.description,
        }


@dataclass
class ClusterGroup:
    cluster_id: str
    source: str
    items: list[ClusterItem]

    @property
    def case_ids(self) -> list[str]:
        return [item.case_id for item in self.items]

    @property
    def size(self) -> int:
        return len(self.items)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "source": self.source,
            "size": self.size,
            "items": [item.to_dict() for item in self.items],
        }


@dataclass
class NodeSummary:
    name: str
    trigger: str
    background: str

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "trigger": self.trigger,
            "background": self.background,
        }


@dataclass
class KnowledgeNode:
    name: str
    trigger: str
    background: str
    children: list["KnowledgeNode"] = field(default_factory=list)
    case_ids: list[str] = field(default_factory=list)
    depth: int = 0
    path: list[str] = field(default_factory=list)

    def to_tree_dict(self) -> dict[str, Any]:
        payload = {
            "name": self.name,
            "trigger": self.trigger,
            "background": self.background,
        }
        if self.children:
            payload["children"] = [child.to_tree_dict() for child in self.children]
        return payload

    def to_debug_dict(self) -> dict[str, Any]:
        payload = {
            "name": self.name,
            "trigger": self.trigger,
            "background": self.background,
            "depth": self.depth,
            "path": self.path,
            "case_ids": self.case_ids,
        }
        if self.children:
            payload["children"] = [child.to_debug_dict() for child in self.children]
        return payload


@dataclass
class NodeBuildResult:
    node: KnowledgeNode
    cases: list[CaseRecord]
    cluster_summaries: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "node": self.node.to_debug_dict(),
            "cases": [case.to_dict() for case in self.cases],
            "cluster_summaries": self.cluster_summaries,
        }
