from __future__ import annotations

from collections import defaultdict

from config import HarnessConfig

from .clustering import cluster_items
from .io_utils import write_json
from .llm import HeuristicLLM, LLMClient
from .models import (
    CaseRecord,
    ClassificationResult,
    ClusterGroup,
    ClusterItem,
    DiscoveryResult,
    KnowledgeNode,
    NodeSummary,
)
from .audit import ErrorAuditCollector
from .reporting import ConsoleReporter
from .utils import TaskOutcome, bounded_gather_outcomes, normalize_software_name, truncate


class L1Builder:
    def __init__(
        self,
        config: HarnessConfig,
        llm_client: LLMClient,
        stage_dir,
        reporter: ConsoleReporter,
        audit: ErrorAuditCollector,
    ):
        self.config = config
        self.llm = llm_client
        self.stage_dir = stage_dir
        self.reporter = reporter
        self.audit = audit
        self.fallback_llm = HeuristicLLM(config)

    async def build(
        self,
        cases: list[CaseRecord],
        seed_nodes: list[KnowledgeNode],
    ) -> KnowledgeNode:
        self.reporter.section(
            "Stage 1: Build L1",
            f"cases={len(cases)} seed_l1={len(seed_nodes)}",
        )
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
        self.reporter.info(
            f"L1 classification complete: matched={len(cases) - len(unmatched_cases)} unmatched={len(unmatched_cases)}"
        )

        discoveries = await self._discover_cases(unmatched_cases)
        write_json(
            self.stage_dir / "02_new_category_discovery.json",
            [item.to_dict() for item in discoveries],
        )
        if discoveries:
            self.reporter.info(f"Discovered {len(discoveries)} unmatched case summaries")
        else:
            self.reporter.info("No unmatched cases, skipping new-category discovery")

        cases_by_id = {case.case_id: case for case in cases}
        new_large_nodes, others_children, cluster_debug, node_debug = await self._build_unmatched_nodes(
            discoveries,
            cases_by_id,
        )
        write_json(self.stage_dir / "03_candidate_clusters.json", cluster_debug)
        write_json(self.stage_dir / "04_candidate_nodes.json", node_debug)
        self.reporter.info(
            f"Candidate node summary complete: new_l1={len(new_large_nodes)} others_children={len(others_children)}"
        )

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
            self._set_subtree_location(node, depth=1, path=["Root", node.name])
            all_children.append(node)

        if others_children:
            others_case_ids: list[str] = []
            for child in others_children:
                others_case_ids.extend(child.case_ids)
                self._set_subtree_location(child, depth=2, path=["Root", "其他", child.name])
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
        self.reporter.info(f"Initial root built with {len(root.children)} L1 children")
        return root

    async def _build_unmatched_nodes(
        self,
        discoveries: list[DiscoveryResult],
        cases_by_id: dict[str, CaseRecord],
    ) -> tuple[list[KnowledgeNode], list[KnowledgeNode], dict[str, object], dict[str, object]]:
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

        new_large_nodes: list[KnowledgeNode] = []
        others_children: list[KnowledgeNode] = []
        cluster_debug: dict[str, object] = {
            "non_software_clusters": [],
            "software_name_groups": [],
            "software_function_clusters": [],
        }
        node_debug: dict[str, object] = {
            "non_software_nodes": [],
            "direct_software_l1_nodes": [],
            "software_small_nodes": [],
            "software_big_nodes": [],
        }

        non_software_clusters = await self._cluster_non_software(non_software_items)
        cluster_debug["non_software_clusters"] = [group.to_dict() for group in non_software_clusters]
        self.reporter.info(f"Generated {len(non_software_clusters)} candidate clusters")
        non_software_nodes = await self._clusters_to_nodes(non_software_clusters, cases_by_id)
        node_debug["non_software_nodes"] = [entry for entry in non_software_nodes["debug"]]
        new_large_nodes.extend(non_software_nodes["new_l1"])
        others_children.extend(non_software_nodes["others"])

        software_result = await self._build_software_nodes(software_items, cases_by_id)
        cluster_debug["software_name_groups"] = software_result["cluster_debug"]["software_name_groups"]
        cluster_debug["software_function_clusters"] = software_result["cluster_debug"]["software_function_clusters"]
        node_debug["direct_software_l1_nodes"] = software_result["node_debug"]["direct_software_l1_nodes"]
        node_debug["software_small_nodes"] = software_result["node_debug"]["software_small_nodes"]
        node_debug["software_big_nodes"] = software_result["node_debug"]["software_big_nodes"]
        new_large_nodes.extend(software_result["new_l1"])
        others_children.extend(software_result["others"])
        self.reporter.info(
            f"Generated {len(non_software_clusters) + len(software_result['all_clusters'])} candidate clusters"
        )
        return new_large_nodes, others_children, cluster_debug, node_debug

    async def _classify_cases(
        self,
        cases: list[CaseRecord],
        seed_nodes: list[KnowledgeNode],
    ) -> list[ClassificationResult]:
        factories = [
            (lambda case=case: self.llm.classify_case(case, seed_nodes))
            for case in cases
        ]
        outcomes = await bounded_gather_outcomes(
            factories,
            self.config.llm.concurrency,
            reporter=self.reporter,
            progress_label="L1 case classification",
        )
        results: list[ClassificationResult] = []
        for case, outcome in zip(cases, outcomes):
            results.append(await self._resolve_classification_outcome(case, seed_nodes, outcome))
        return results

    async def _discover_cases(self, cases: list[CaseRecord]) -> list[DiscoveryResult]:
        if not cases:
            return []
        factories = [(lambda case=case: self.llm.discover_case(case)) for case in cases]
        outcomes = await bounded_gather_outcomes(
            factories,
            self.config.llm.concurrency,
            reporter=self.reporter,
            progress_label="Discover unmatched cases",
        )
        results: list[DiscoveryResult] = []
        for case, outcome in zip(cases, outcomes):
            results.append(await self._resolve_discovery_outcome(case, outcome))
        return results

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

    async def _clusters_to_nodes(
        self,
        clusters: list[ClusterGroup],
        cases_by_id: dict[str, CaseRecord],
    ) -> dict[str, object]:
        new_l1: list[KnowledgeNode] = []
        others: list[KnowledgeNode] = []
        debug_entries: list[dict[str, object]] = []
        progress = self.reporter.progress(len(clusters), "Summarize candidate clusters")
        for cluster_group in clusters:
            cluster_cases = [cases_by_id[case_id] for case_id in cluster_group.case_ids]
            summary = await self._summarize_candidate_cluster(cluster_group, cluster_cases)
            node = KnowledgeNode(
                name=summary.name,
                trigger=summary.trigger,
                background=summary.background,
                case_ids=cluster_group.case_ids,
            )
            debug_entries.append(
                {
                    "cluster": cluster_group.to_dict(),
                    "summary": summary.to_dict(),
                    "case_count": len(cluster_group.case_ids),
                    "placement": "l1"
                    if len(cluster_group.case_ids) >= self.config.pipeline.new_l1_min_cases
                    else "others",
                }
            )
            if len(cluster_group.case_ids) >= self.config.pipeline.new_l1_min_cases:
                new_l1.append(node)
            else:
                others.append(node)
            progress.update(1)
        progress.close()
        return {
            "new_l1": new_l1,
            "others": others,
            "debug": debug_entries,
        }

    async def _cluster_non_software(self, items: list[ClusterItem]) -> list[ClusterGroup]:
        if not items:
            return []
        return await cluster_items(
            items,
            self.config.cluster,
            reporter=self.reporter,
            label="non-software L1 discovery",
        )

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
            grouped = await cluster_items(
                group_items,
                self.config.cluster,
                reporter=self.reporter,
                label=f"software L1 discovery:{software_name}",
            )
            clusters.extend(grouped)
        return clusters

    async def _build_software_nodes(
        self,
        items: list[ClusterItem],
        cases_by_id: dict[str, CaseRecord],
    ) -> dict[str, object]:
        if not items:
            return {
                "new_l1": [],
                "others": [],
                "all_clusters": [],
                "cluster_debug": {
                    "software_name_groups": [],
                    "software_function_clusters": [],
                },
                "node_debug": {
                    "direct_software_l1_nodes": [],
                    "software_small_nodes": [],
                    "software_big_nodes": [],
                },
            }

        grouped_by_name = self._group_software_items_by_name(items)
        software_name_debug = [
            {
                "software_name": software_name,
                "case_ids": [item.case_id for item in group_items],
                "size": len(group_items),
                "descriptions": [item.description for item in group_items],
            }
            for software_name, group_items in grouped_by_name.items()
        ]

        direct_l1_nodes: list[KnowledgeNode] = []
        others_children: list[KnowledgeNode] = []
        all_clusters: list[ClusterGroup] = []
        direct_l1_debug: list[dict[str, object]] = []
        small_node_debug: list[dict[str, object]] = []
        big_node_debug: list[dict[str, object]] = []

        small_function_items: list[ClusterItem] = []
        small_node_map: dict[str, KnowledgeNode] = {}

        for index, (software_name, group_items) in enumerate(grouped_by_name.items(), start=1):
            group_cluster = ClusterGroup(
                cluster_id=f"software_group_{software_name or index}",
                source="software_name_group",
                items=group_items,
            )
            all_clusters.append(group_cluster)
            cluster_cases = [cases_by_id[item.case_id] for item in group_items]

            if len(group_items) >= self.config.pipeline.new_l1_min_cases:
                summary = await self._summarize_candidate_cluster(group_cluster, cluster_cases)
                node = KnowledgeNode(
                    name=summary.name,
                    trigger=summary.trigger,
                    background=summary.background,
                    case_ids=[item.case_id for item in group_items],
                )
                direct_l1_nodes.append(node)
                direct_l1_debug.append(
                    {
                        "software_name": software_name,
                        "cluster": group_cluster.to_dict(),
                        "summary": summary.to_dict(),
                        "placement": "l1_direct",
                    }
                )
                continue

            summary = await self._summarize_candidate_cluster(group_cluster, cluster_cases)
            small_node = KnowledgeNode(
                name=summary.name,
                trigger=summary.trigger,
                background=summary.background,
                case_ids=[item.case_id for item in group_items],
            )
            pseudo_id = f"software_function_{software_name or index}"
            small_node_map[pseudo_id] = small_node
            small_function_items.append(
                ClusterItem(
                    case_id=pseudo_id,
                    text=self._node_cluster_text(summary),
                    source="software_function",
                    software_name=software_name,
                    description=summary.name,
                )
            )
            small_node_debug.append(
                {
                    "software_name": software_name,
                    "cluster": group_cluster.to_dict(),
                    "summary": summary.to_dict(),
                    "placement": "await_big_node",
                }
            )

        function_clusters = await cluster_items(
            small_function_items,
            self.config.cluster,
            reporter=self.reporter,
            label="software function regrouping",
        )
        all_clusters.extend(function_clusters)

        progress = self.reporter.progress(len(function_clusters), "Summarize software big nodes")
        for function_cluster in function_clusters:
            child_nodes = [small_node_map[pseudo_id] for pseudo_id in function_cluster.case_ids]
            big_case_ids = list(
                dict.fromkeys(
                    case_id
                    for child in child_nodes
                    for case_id in child.case_ids
                )
            )
            cluster_cases = [cases_by_id[case_id] for case_id in big_case_ids]
            summary = await self._summarize_candidate_cluster(function_cluster, cluster_cases)
            big_node = KnowledgeNode(
                name=summary.name,
                trigger=summary.trigger,
                background=summary.background,
                children=child_nodes,
                case_ids=big_case_ids,
            )
            placement = "l1" if len(big_case_ids) >= self.config.pipeline.new_l1_min_cases else "others"
            if placement == "l1":
                direct_l1_nodes.append(big_node)
            else:
                others_children.append(big_node)
            big_node_debug.append(
                {
                    "cluster": function_cluster.to_dict(),
                    "summary": summary.to_dict(),
                    "placement": placement,
                    "aggregated_case_ids": big_case_ids,
                    "children": [child.to_debug_dict() for child in child_nodes],
                }
            )
            progress.update(1)
        progress.close()

        return {
            "new_l1": direct_l1_nodes,
            "others": others_children,
            "all_clusters": all_clusters,
            "cluster_debug": {
                "software_name_groups": software_name_debug,
                "software_function_clusters": [group.to_dict() for group in function_clusters],
            },
            "node_debug": {
                "direct_software_l1_nodes": direct_l1_debug,
                "software_small_nodes": small_node_debug,
                "software_big_nodes": big_node_debug,
            },
        }

    def _group_software_items_by_name(self, items: list[ClusterItem]) -> dict[str, list[ClusterItem]]:
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
        return grouped_by_name

    def _node_cluster_text(self, summary: NodeSummary) -> str:
        return f"{summary.name}\n{summary.trigger}\n{summary.background}".strip()

    def _set_subtree_location(
        self,
        node: KnowledgeNode,
        depth: int,
        path: list[str],
    ) -> None:
        node.depth = depth
        node.path = path
        for child in node.children:
            self._set_subtree_location(
                child,
                depth=depth + 1,
                path=[*path, child.name],
            )

    async def _resolve_classification_outcome(
        self,
        case: CaseRecord,
        seed_nodes: list[KnowledgeNode],
        outcome: TaskOutcome[ClassificationResult],
    ) -> ClassificationResult:
        if outcome.ok and outcome.value is not None:
            return outcome.value

        assert outcome.error is not None
        self.reporter.warn(f"L1 classification failed for {case.case_id}, using heuristic fallback")
        try:
            fallback = await self.fallback_llm.classify_case(case, seed_nodes)
            fallback.reason = f"{fallback.reason}；主LLM失败后使用heuristic回退"
            self.audit.record_case_failure(
                stage="l1_classification",
                case=case,
                error=outcome.error,
                fallback="heuristic",
                fallback_succeeded=True,
            )
            return fallback
        except Exception as fallback_error:  # noqa: BLE001
            self.audit.record_case_failure(
                stage="l1_classification",
                case=case,
                error=outcome.error,
                fallback="heuristic",
                fallback_succeeded=False,
                details={"fallback_error": str(fallback_error)},
            )
            return ClassificationResult(
                case_id=case.case_id,
                belongs=False,
                category_name=None,
                reason="主LLM与heuristic回退均失败，按未匹配案例处理",
            )

    async def _resolve_discovery_outcome(
        self,
        case: CaseRecord,
        outcome: TaskOutcome[DiscoveryResult],
    ) -> DiscoveryResult:
        if outcome.ok and outcome.value is not None:
            return outcome.value

        assert outcome.error is not None
        self.reporter.warn(f"New-category discovery failed for {case.case_id}, using heuristic fallback")
        try:
            fallback = await self.fallback_llm.discover_case(case)
            self.audit.record_case_failure(
                stage="l1_discovery",
                case=case,
                error=outcome.error,
                fallback="heuristic",
                fallback_succeeded=True,
            )
            return fallback
        except Exception as fallback_error:  # noqa: BLE001
            self.audit.record_case_failure(
                stage="l1_discovery",
                case=case,
                error=outcome.error,
                fallback="heuristic",
                fallback_succeeded=False,
                details={"fallback_error": str(fallback_error)},
            )
            return DiscoveryResult(
                case_id=case.case_id,
                software_name="",
                description=truncate(case.content, 60),
            )

    async def _summarize_candidate_cluster(
        self,
        cluster_group: ClusterGroup,
        cluster_cases: list[CaseRecord],
    ) -> NodeSummary:
        try:
            return await self.llm.summarize_candidate_cluster(cluster_group, cluster_cases)
        except Exception as error:  # noqa: BLE001
            self.reporter.warn(
                f"Candidate cluster summary failed for {cluster_group.cluster_id}, using heuristic fallback"
            )
            try:
                summary = await self.fallback_llm.summarize_candidate_cluster(cluster_group, cluster_cases)
                self.audit.record_item_failure(
                    stage="l1_candidate_cluster_summary",
                    item_type="cluster",
                    item_id=cluster_group.cluster_id,
                    error=error,
                    fallback="heuristic",
                    fallback_succeeded=True,
                    details={"case_ids": cluster_group.case_ids},
                )
                return summary
            except Exception as fallback_error:  # noqa: BLE001
                self.audit.record_item_failure(
                    stage="l1_candidate_cluster_summary",
                    item_type="cluster",
                    item_id=cluster_group.cluster_id,
                    error=error,
                    fallback="heuristic",
                    fallback_succeeded=False,
                    details={
                        "case_ids": cluster_group.case_ids,
                        "fallback_error": str(fallback_error),
                    },
                )
                name = truncate(cluster_cases[0].case_name, 24) if cluster_cases else cluster_group.cluster_id
                return NodeSummary(
                    name=name,
                    trigger=f"当问题表现为{name}相关场景时考虑该类别",
                    background=f"{name}为回退生成的候选类别，请结合原始案例进一步校验。",
                )
