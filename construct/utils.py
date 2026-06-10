from __future__ import annotations

import asyncio
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Awaitable, Callable, Generic, Iterable, TypeVar


T = TypeVar("T")

if TYPE_CHECKING:
    from .reporting import ConsoleReporter


@dataclass
class TaskOutcome(Generic[T]):
    index: int
    value: T | None = None
    error: Exception | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

ENGLISH_STOP_WORDS = {
    "appdata",
    "wifi",
    "sqlite",
    "http",
    "https",
    "windows",
    "direct3d",
    "opengl",
    "large",
    "file",
    "files",
    "office",
    "manager",
    "credential",
    "microsoftoffice16",
    "publickey",
    "excel",
    "storage",
}

CHINESE_SEPARATORS = re.compile(r"[，。；：、（）“”\"'《》【】\[\]\s\-_/]+")
ENGLISH_TOKENS = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip())


def normalize_software_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def extract_english_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    for token in ENGLISH_TOKENS.findall(text):
        normalized = token.strip()
        if normalized.lower() in ENGLISH_STOP_WORDS:
            continue
        candidates.append(normalized)
    return candidates


def split_mixed_tokens(text: str) -> list[str]:
    chinese_parts = [part for part in CHINESE_SEPARATORS.split(text) if part]
    english_parts = [token.lower() for token in ENGLISH_TOKENS.findall(text)]
    return chinese_parts + english_parts


def top_terms(texts: Iterable[str], limit: int = 3) -> list[str]:
    counter: Counter[str] = Counter()
    for text in texts:
        for token in split_mixed_tokens(text):
            if len(token) <= 1:
                continue
            counter[token] += 1
    return [term for term, _ in counter.most_common(limit)]


def slugify(text: str, fallback: str) -> str:
    lowered = text.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", lowered).strip("-")
    return slug or fallback


def truncate(text: str, limit: int = 80) -> str:
    clean = normalize_text(text)
    if len(clean) <= limit:
        return clean
    return f"{clean[: limit - 3]}..."


async def bounded_gather(
    factories: list[Callable[[], Awaitable[T]]],
    concurrency: int,
    reporter: "ConsoleReporter | None" = None,
    progress_label: str = "",
) -> list[T]:
    if not factories:
        return []

    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def run(index: int, factory: Callable[[], Awaitable[T]]) -> tuple[int, T]:
        async with semaphore:
            return index, await factory()

    tasks = [asyncio.create_task(run(index, factory)) for index, factory in enumerate(factories)]
    results: list[T | None] = [None] * len(tasks)
    progress = reporter.progress(len(tasks), progress_label) if reporter and progress_label else None

    try:
        for task in asyncio.as_completed(tasks):
            index, value = await task
            results[index] = value
            if progress is not None:
                progress.update(1)
    except Exception:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    finally:
        if progress is not None:
            progress.close()

    return [value for value in results if value is not None]


async def bounded_gather_outcomes(
    factories: list[Callable[[], Awaitable[T]]],
    concurrency: int,
    reporter: "ConsoleReporter | None" = None,
    progress_label: str = "",
) -> list[TaskOutcome[T]]:
    if not factories:
        return []

    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def run(index: int, factory: Callable[[], Awaitable[T]]) -> TaskOutcome[T]:
        async with semaphore:
            try:
                return TaskOutcome(index=index, value=await factory())
            except Exception as exc:  # noqa: BLE001
                return TaskOutcome(index=index, error=exc)

    tasks = [asyncio.create_task(run(index, factory)) for index, factory in enumerate(factories)]
    results: list[TaskOutcome[T] | None] = [None] * len(tasks)
    progress = reporter.progress(len(tasks), progress_label) if reporter and progress_label else None

    try:
        for task in asyncio.as_completed(tasks):
            outcome = await task
            results[outcome.index] = outcome
            if progress is not None:
                progress.update(1)
    finally:
        if progress is not None:
            progress.close()

    return [result for result in results if result is not None]
