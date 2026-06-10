from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .models import CaseRecord


def _timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


@dataclass
class AuditEntry:
    stage: str
    item_type: str
    item_id: str
    message: str
    exception_type: str
    fallback: str = "none"
    fallback_succeeded: bool = False
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=_timestamp)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "stage": self.stage,
            "item_type": self.item_type,
            "item_id": self.item_id,
            "message": self.message,
            "exception_type": self.exception_type,
            "fallback": self.fallback,
            "fallback_succeeded": self.fallback_succeeded,
            "details": self.details,
        }


class ErrorAuditCollector:
    def __init__(self) -> None:
        self.entries: list[AuditEntry] = []
        self.case_failures: dict[str, dict[str, Any]] = {}

    def record_case_failure(
        self,
        stage: str,
        case: CaseRecord,
        error: Exception,
        fallback: str = "none",
        fallback_succeeded: bool = False,
        details: dict[str, Any] | None = None,
    ) -> None:
        entry = AuditEntry(
            stage=stage,
            item_type="case",
            item_id=case.case_id,
            message=str(error),
            exception_type=type(error).__name__,
            fallback=fallback,
            fallback_succeeded=fallback_succeeded,
            details={
                "case_name": case.case_name,
                **(details or {}),
            },
        )
        self.entries.append(entry)

        failure = self.case_failures.setdefault(
            case.case_id,
            {
                "case_id": case.case_id,
                "case_name": case.case_name,
                "failure_count": 0,
                "stages": [],
                "events": [],
            },
        )
        failure["failure_count"] += 1
        failure["stages"].append(stage)
        failure["events"].append(
            {
                "timestamp": entry.timestamp,
                "stage": stage,
                "message": entry.message,
                "exception_type": entry.exception_type,
                "fallback": fallback,
                "fallback_succeeded": fallback_succeeded,
            }
        )

    def record_item_failure(
        self,
        stage: str,
        item_type: str,
        item_id: str,
        error: Exception,
        fallback: str = "none",
        fallback_succeeded: bool = False,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.entries.append(
            AuditEntry(
                stage=stage,
                item_type=item_type,
                item_id=item_id,
                message=str(error),
                exception_type=type(error).__name__,
                fallback=fallback,
                fallback_succeeded=fallback_succeeded,
                details=details or {},
            )
        )

    def summary(self) -> dict[str, Any]:
        stages = Counter(entry.stage for entry in self.entries)
        item_types = Counter(entry.item_type for entry in self.entries)
        fallback_successes = sum(1 for entry in self.entries if entry.fallback_succeeded)
        return {
            "total_errors": len(self.entries),
            "failed_case_count": len(self.case_failures),
            "fallback_success_count": fallback_successes,
            "errors_by_stage": dict(stages),
            "errors_by_item_type": dict(item_types),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary(),
            "entries": [entry.to_dict() for entry in self.entries],
        }

    def failed_cases_payload(self) -> dict[str, Any]:
        return {
            "summary": {
                "failed_case_count": len(self.case_failures),
                "failed_event_count": sum(item["failure_count"] for item in self.case_failures.values()),
            },
            "cases": list(self.case_failures.values()),
        }
