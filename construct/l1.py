from __future__ import annotations

from collections import defaultdict

from config import HarnessConfig

from .clustering import cluster_items
from .io_utils import write_json
from .llm import LLMClient
from .models import (
    CaseRecord,
    ClassificationResult,
    ClusterGroup,
    ClusterItem,
    DiscoveryResult,
    KnowledgeNode,
)
from .utils import bounded_gather, normalize_software_name


class L1Builder:
    def __init__(self, config: HarnessConfig, llm_client: LLMClient, stage_dir):
        self.config = config
        self.llm = llm_client
        self.stage_dir = stage_dir

    async def build(
        self,
        cases: list[CaseRecord],
        seed_nodes: list[KnowledgeNode],
    ) -> KnowledgeNode:
        classification_results = await self._classify_cases(cases, seed_nodes)
        write_json(
            self.stage_dir / "01_l1_classification.json",
            [result.to_dict() for result in classification_results],
        )

        matched_cases: dict[str, list[str]] = defaultdict(list)
        unmatched_cases: list[CaseRecord] = []
        seed_name_map = {node.name: node for node in seed_nodes}
        for result in classification_results:
            if result.belongs and result.category_name in seed_name_map:
                matched_cases[result.category_name].append(result.case_id)
            else:
                case = next(case for case in cases if case.case_id == result.case_id)
                unmatched_cases.append(case)

        discoveries = await self._discover_cases(unmatched_cases)
        write_json(
            self.stage_dir / "02_new_category_discovery.json",
            [item.to_dict() for item in discoveries],
        )

        clusters = await self._build_candidate_clusters(discoveries)
        write_json(
            self.stage_dir / "03_candidate_clusters.json",
            [group.to_dict() for group in clusters],
        )

        new_large_nodes: list[KnowledgeNode] = []
        others_children: list[KnowledgeNode] = []
        candidate_summaries: list[dict[str, object]] = []

        cases_by_id = {case.case_id: case for case in cases}
        for cluster_group in clusters:
            cluster_cases = [cases_by_id[case_id] for case_id in cluster_group.case_ids]
            summary = await self.llm.summarize_candidate_cluster(cluster_group, cluster_cases)
            candidate_summaries.append(
                {
                    "cluster": cluster_group.to_dict(),
                    "summary": summary.to_dict(),
                }
            )
            node = KnowledgeNode(
                name=summary.name,
                trigger=summary.trigger,
                background=summary.background,
                case_ids=cluster_group.case_ids,
            )
            if cluster_group.size >= self.config.pipeline.new_l1_min_cases:
                node.depth = 1
                new_large_nodes.append(node)
            else:
                node.depth = 2
                others_children.append(node)

        write_json(self.stage_dir / "04_candidate_nodes.json", candidate_summaries)

        root = KnowledgeNode(
            name="Root",
            trigger="知识树根节点",
            background="基于案例库构建的层级化知识结构根节点。",
            depth=0,
            path=["Root"],
        )

        all_children: list[KnowledgeNode] = []
        for seed_node in seed_nodes:
            child = KnowledgeNode(
                name=seed_node.name,
                trigger=seed_node.trigger,
                background=seed_node.background,
                case_ids=matched_cases.get(seed_node.name, []),
                depth=1,
                path=["Root", seed_node.name],
            )
            all_children.append(child)

        for node in new_large_nodes:
            node.path = ["Root", node.name]
            all_children.append(node)

        if others_children:
            others_case_ids: list[str] = []
            for child in others_children:
                child.path = ["Root", "其他", child.name]
                others_case_ids.extend(child.case_ids)
            others_node = KnowledgeNode(
                name="其他",
                trigger="当案例无法稳定归入已有L1且聚类规模较小时考虑该类别",
                background="其他类别用于承接当前规模较小但语义相近的长尾案例簇。",
                children=others_children,
                case_ids=others_case_ids,
                depth=1,
                path=["Root", "其他"],
            )
            all_children.append(others_node)

        root.children = all_children
        write_json(self.stage_dir / "05_initial_root.json", root.to_debug_dict())
        return root

    async def _classify_cases(
        self,
        cases: list[CaseRecord],
        seed_nodes: list[KnowledgeNode],
    ) -> list[ClassificationResult]:
        factories = [
            (lambda case=case: self.llm.classify_case(case, seed_nodes))
            for case in cases
        ]
        return await bounded_gather(factories, self.config.llm.concurrency)

    async def _discover_cases(self, cases: list[CaseRecord]) -> list[DiscoveryResult]:
        if not cases:
            return []
        factories = [(lambda case=case: self.llm.discover_case(case)) for case in cases]
        return await bounded_gather(factories, self.config.llm.concurrency)

    async def _build_candidate_clusters(
        self,
        discoveries: list[DiscoveryResult],
    ) -> list[ClusterGroup]:
        if not discoveries:
            return []

        software_items: list[ClusterItem] = []
        non_software_items: list[ClusterItem] = []
        for discovery in discoveries:
            item = ClusterItem(
                case_id=discovery.case_id,
                text=discovery.description,
                source="software" if discovery.software_name else "non_software",
                software_name=discovery.software_name,
                description=discovery.description,
            )
            if discovery.software_name.strip():
                software_items.append(item)
            else:
                non_software_items.append(item)

        clusters: list[ClusterGroup] = []
        clusters.extend(await self._cluster_non_software(non_software_items))
        clusters.extend(await self._cluster_software(software_items))
        return clusters

    async def _cluster_non_software(self, items: list[ClusterItem]) -> list[ClusterGroup]:
        if not items:
            return []
        return await cluster_items(items, self.config.cluster)

    async def _cluster_software(self, items: list[ClusterItem]) -> list[ClusterGroup]:
        if not items:
            return []

        grouped_by_name: dict[str, list[ClusterItem]] = defaultdict(list)
        canonical_names: list[str] = []
        for item in sorted(items, key=lambda current: len(current.software_name or current.text)):
            software_name = item.software_name.strip()
            normalized = normalize_software_name(software_name)
            if not normalized:
                grouped_by_name[item.case_id].append(item)
                continue

            matched_canonical = None
            for canonical in canonical_names:
                probe_len = max(
                    1,
                    min(
                        self.config.pipeline.software_alias_min_match,
                        len(canonical),
                        len(normalized),
                    ),
                )
                canonical_probe = canonical[:probe_len]
                normalized_probe = normalized[:probe_len]
                if canonical_probe in normalized or normalized_probe in canonical:
                    matched_canonical = canonical
                    break

            if matched_canonical is None:
                canonical_names.append(normalized)
                matched_canonical = normalized

            grouped_by_name[matched_canonical].append(item)

        clusters: list[ClusterGroup] = []
        for software_name, group_items in grouped_by_name.items():
            if len(group_items) >= self.config.pipeline.new_l1_min_cases:
                clusters.append(
                    ClusterGroup(
                        cluster_id=f"software_group_{software_name or group_items[0].case_id}",
                        source="software",
                        items=group_items,
                    )
                )
                continue
            if len(group_items) == 1:
                clusters.append(
                    ClusterGroup(
                        cluster_id=f"software_{software_name or group_items[0].case_id}",
                        source="software",
                        items=group_items,
                    )
                )
                continue
            grouped = await cluster_items(group_items, self.config.cluster)
            clusters.extend(grouped)
        return clusters
