from __future__ import annotations

import asyncio
import re
from collections import Counter
from pathlib import Path
from typing import Awaitable, Callable, Iterable, TypeVar


T = TypeVar("T")

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
) -> list[T]:
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def run(factory: Callable[[], Awaitable[T]]) -> T:
        async with semaphore:
            return await factory()

    return await asyncio.gather(*(run(factory) for factory in factories))
