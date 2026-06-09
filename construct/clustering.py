from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Iterable

from cluster import cluster as external_cluster
from config import ClusterSettings

from .models import ClusterGroup, ClusterItem
from .utils import split_mixed_tokens

if TYPE_CHECKING:
    from .reporting import ConsoleReporter


async def cluster_items(
    items: list[ClusterItem],
    settings: ClusterSettings,
    reporter: "ConsoleReporter | None" = None,
    label: str = "",
) -> list[ClusterGroup]:
    if not items:
        return []
    if len(items) == 1:
        return [ClusterGroup(cluster_id="cluster_0", source=items[0].source, items=items[:])]

    if reporter is not None:
        target = label or items[0].source
        reporter.info(
            f"Clustering {len(items)} items for {target} with method={settings.method}"
        )

    assignments = await _cluster_assignments([item.text for item in items], settings)
    if not assignments or len(assignments) != len(items):
        if reporter is not None:
            reporter.warn("External cluster backend unavailable, using local fallback clustering")
        assignments = _local_cluster_assignments([item.text for item in items], settings)

    grouped: dict[int, list[ClusterItem]] = defaultdict(list)
    for item, cluster_id in zip(items, assignments):
        grouped[cluster_id].append(item)

    result: list[ClusterGroup] = []
    ordered_cluster_ids = sorted(grouped.keys(), key=lambda value: (value == -1, value))
    noise_index = 0
    for cluster_id in ordered_cluster_ids:
        cluster_items_list = grouped[cluster_id]
        if cluster_id == -1:
            for item in cluster_items_list:
                result.append(
                    ClusterGroup(
                        cluster_id=f"noise_{noise_index}",
                        source=item.source,
                        items=[item],
                    )
                )
                noise_index += 1
            continue
        result.append(
            ClusterGroup(
                cluster_id=f"cluster_{cluster_id}",
                source=cluster_items_list[0].source,
                items=cluster_items_list,
            )
        )
    if reporter is not None:
        target = label or items[0].source
        reporter.info(f"Clustering finished for {target}: {len(result)} clusters")
    return result


async def _cluster_assignments(texts: list[str], settings: ClusterSettings) -> list[int] | None:
    try:
        results = await external_cluster(
            texts=texts,
            cluster_method=settings.method,
            n_clusters=settings.n_clusters,
            min_cluster_size=settings.min_cluster_size,
            embedding_concurrency=settings.embedding_concurrency,
            embedding_model=settings.embedding_model,
            embedding_url=settings.embedding_url,
            embedding_api_key=settings.embedding_api_key,
        )
    except Exception:  # noqa: BLE001
        return None

    if not isinstance(results, list) or len(results) != len(texts):
        return None

    cluster_ids: list[int] = []
    for item in results:
        if not isinstance(item, dict) or "cluster_id" not in item:
            return None
        try:
            cluster_ids.append(int(item["cluster_id"]))
        except (TypeError, ValueError):
            return None
    return cluster_ids


def _local_cluster_assignments(texts: list[str], settings: ClusterSettings) -> list[int]:
    if settings.method == "kmeans":
        return _local_kmeans(texts, settings.n_clusters)
    return _local_density_groups(texts, settings.min_cluster_size, settings.local_similarity_threshold)


def _local_density_groups(
    texts: list[str],
    min_cluster_size: int,
    similarity_threshold: float,
) -> list[int]:
    tokenized = [_token_frequency(text) for text in texts]
    parent = list(range(len(texts)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            similarity = _cosine_similarity(tokenized[i], tokenized[j])
            if similarity >= similarity_threshold:
                union(i, j)

    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(texts)):
        groups[find(index)].append(index)

    assignments = [-1] * len(texts)
    next_cluster = 0
    for indices in groups.values():
        if len(indices) < min_cluster_size:
            continue
        for index in indices:
            assignments[index] = next_cluster
        next_cluster += 1

    return assignments


def _local_kmeans(texts: list[str], n_clusters: int) -> list[int]:
    cluster_count = max(1, min(n_clusters, len(texts)))
    vectors = [_token_frequency(text) for text in texts]
    centroids = [vectors[index].copy() for index in range(cluster_count)]
    assignments = [0] * len(texts)

    for _ in range(8):
        changed = False
        for index, vector in enumerate(vectors):
            best_cluster = 0
            best_score = -1.0
            for cluster_index, centroid in enumerate(centroids):
                score = _cosine_similarity(vector, centroid)
                if score > best_score:
                    best_score = score
                    best_cluster = cluster_index
            if assignments[index] != best_cluster:
                assignments[index] = best_cluster
                changed = True

        if not changed:
            break

        bucketed: dict[int, list[Counter[str]]] = defaultdict(list)
        for index, cluster_index in enumerate(assignments):
            bucketed[cluster_index].append(vectors[index])
        for cluster_index in range(cluster_count):
            if not bucketed[cluster_index]:
                continue
            centroids[cluster_index] = _average_counter(bucketed[cluster_index])

    return assignments


def _token_frequency(text: str) -> Counter[str]:
    return Counter(token for token in split_mixed_tokens(text) if len(token) > 1)


def _average_counter(counters: Iterable[Counter[str]]) -> Counter[str]:
    totals: Counter[str] = Counter()
    count = 0
    for counter in counters:
        totals.update(counter)
        count += 1
    if count == 0:
        return Counter()
    averaged = Counter()
    for key, value in totals.items():
        averaged[key] = value / count
    return averaged


def _cosine_similarity(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    common = set(left) & set(right)
    numerator = sum(float(left[key]) * float(right[key]) for key in common)
    left_norm = math.sqrt(sum(float(value) ** 2 for value in left.values()))
    right_norm = math.sqrt(sum(float(value) ** 2 for value in right.values()))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)
