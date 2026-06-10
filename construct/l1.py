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

        clusters = await self._build_candidate_clusters(discoveries)
        write_json(
            self.stage_dir / "03_candidate_clusters.json",
            [group.to_dict() for group in clusters],
        )
        self.reporter.info(f"Generated {len(clusters)} candidate clusters")

        new_large_nodes: list[KnowledgeNode] = []
        others_children: list[KnowledgeNode] = []
        candidate_summaries: list[dict[str, object]] = []

        cases_by_id = {case.case_id: case for case in cases}
        progress = self.reporter.progress(len(clusters), "Summarize candidate clusters")
        for cluster_group in clusters:
            cluster_cases = [cases_by_id[case_id] for case_id in cluster_group.case_ids]
            summary = await self._summarize_candidate_cluster(cluster_group, cluster_cases)
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
            progress.update(1)
        progress.close()

        write_json(self.stage_dir / "04_candidate_nodes.json", candidate_summaries)
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
        self.reporter.info(f"Initial root built with {len(root.children)} L1 children")
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
