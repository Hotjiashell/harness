from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from config import HarnessConfig

from .audit import ErrorAuditCollector
from .clustering import cluster_items
from .io_utils import write_json
from .llm import HeuristicLLM, LLMClient
from .models import CaseRecord, ClusterGroup, ClusterItem, ClusterNodeSummary, KnowledgeNode
from .reporting import ConsoleReporter
from .utils import TaskOutcome, bounded_gather_outcomes, slugify, truncate


class RecursiveTreeBuilder:
    def __init__(
        self,
        config: HarnessConfig,
        llm_client: LLMClient,
        stage_dir: Path,
        cases_by_id: dict[str, CaseRecord],
        reporter: ConsoleReporter,
        audit: ErrorAuditCollector,
    ) -> None:
        self.config = config
        self.llm = llm_client
        self.stage_dir = stage_dir
        self.cases_by_id = cases_by_id
        self.reporter = reporter
        self.audit = audit
        self.fallback_llm = HeuristicLLM(config)

    async def build(self, root: KnowledgeNode) -> None:
        self.reporter.section("Stage 2: Build L2/L3")
        for child in root.children:
            await self._expand_node(child)

    async def _expand_node(self, node: KnowledgeNode) -> None:
        if node.depth >= self.config.pipeline.max_depth:
            return

        if node.children:
            for child in node.children:
                await self._expand_node(child)
            return

        cases = [self.cases_by_id[case_id] for case_id in node.case_ids if case_id in self.cases_by_id]
        if len(cases) < self.config.pipeline.min_cases_to_split:
            return

        self.reporter.info(
            f"Expanding node {' > '.join(node.path)} depth={node.depth} cases={len(cases)}"
        )

        node_stage_dir = self.stage_dir / "tree" / self._node_stage_name(node)
        case_summaries = await self._summarize_cases(node, cases)
        write_json(
            node_stage_dir / "01_case_summaries.json",
            [
                {
                    "case_id": case.case_id,
                    "case_name": case.case_name,
                    "subproblem": summary,
                }
                for case, summary in zip(cases, case_summaries)
            ],
        )

        clusters = await cluster_items(
            [
                ClusterItem(case_id=case.case_id, text=summary, source="child")
                for case, summary in zip(cases, case_summaries)
            ],
            self.config.cluster,
            reporter=self.reporter,
            label=f"tree:{' > '.join(node.path)}",
        )
        write_json(node_stage_dir / "02_clusters.json", [group.to_dict() for group in clusters])

        non_empty_clusters = [group for group in clusters if group.case_ids]
        if len(non_empty_clusters) <= 1 and non_empty_clusters and len(non_empty_clusters[0].case_ids) == len(cases):
            self.reporter.info(f"Node {' > '.join(node.path)} kept as leaf after clustering")
            return

        children = await self._build_children(node, clusters, cases, case_summaries)
        if not children:
            return
        node.children = children
        write_json(node_stage_dir / "03_children.json", [child.to_debug_dict() for child in children])
        self.reporter.info(
            f"Expanded node {' > '.join(node.path)} into {len(children)} children"
        )

        for child in node.children:
            await self._expand_node(child)

    async def _summarize_cases(self, parent: KnowledgeNode, cases: list[CaseRecord]) -> list[str]:
        factories = [
            (lambda case=case: self.llm.summarize_case_under_parent(parent, case))
            for case in cases
        ]
        outcomes = await bounded_gather_outcomes(
            factories,
            self.config.llm.concurrency,
            reporter=self.reporter,
            progress_label=f"Summarize cases for {parent.name}",
        )
        summaries: list[str] = []
        for case, outcome in zip(cases, outcomes):
            summaries.append(await self._resolve_case_summary_outcome(parent, case, outcome))
        return summaries

    async def _build_children(
        self,
        parent: KnowledgeNode,
        clusters: list[ClusterGroup],
        cases: list[CaseRecord],
        case_summaries: list[str],
    ) -> list[KnowledgeNode]:
        summaries_by_case_id = {
            case.case_id: summary
            for case, summary in zip(cases, case_summaries)
        }
        cases_by_id = {case.case_id: case for case in cases}

        cluster_payloads: list[tuple[ClusterGroup, list[CaseRecord], list[str]]] = []
        for cluster_group in clusters:
            cluster_cases = [cases_by_id[case_id] for case_id in cluster_group.case_ids]
            if not cluster_cases:
                continue
            cluster_summaries = [summaries_by_case_id[case_id] for case_id in cluster_group.case_ids]
            cluster_payloads.append((cluster_group, cluster_cases, cluster_summaries))

        factories = [
            (
                lambda cluster_group=cluster_group, cluster_cases=cluster_cases, cluster_summaries=cluster_summaries:
                self.llm.summarize_child_cluster_partitions(
                    parent,
                    cluster_group,
                    cluster_cases,
                    cluster_summaries,
                )
            )
            for cluster_group, cluster_cases, cluster_summaries in cluster_payloads
        ]
        outcomes = await bounded_gather_outcomes(
            factories,
            self.config.llm.concurrency,
            reporter=self.reporter,
            progress_label=f"Summarize child clusters for {parent.name}",
        )
        summaries: list[list[ClusterNodeSummary]] = []
        for (cluster_group, cluster_cases, cluster_summaries), outcome in zip(cluster_payloads, outcomes):
            summaries.append(
                await self._resolve_child_cluster_summary_outcome(
                    parent,
                    cluster_group,
                    cluster_cases,
                    cluster_summaries,
                    outcome,
                )
            )

        results: list[KnowledgeNode] = []
        for (cluster_group, _, _), summary_group in zip(cluster_payloads, summaries):
            self._log_child_cluster_partition_result(parent, cluster_group, summary_group)
            for summary in summary_group:
                child = KnowledgeNode(
                    name=summary.name,
                    trigger=summary.trigger,
                    background=summary.background,
                    case_ids=summary.item_ids,
                    depth=parent.depth + 1,
                    path=[*parent.path, summary.name],
                )
                results.append(child)

        deduplicated = self._deduplicate_names(results)
        return deduplicated

    def _deduplicate_names(self, nodes: list[KnowledgeNode]) -> list[KnowledgeNode]:
        seen: dict[str, int] = defaultdict(int)
        for node in nodes:
            seen[node.name] += 1
            if seen[node.name] > 1:
                node.name = f"{node.name}_{seen[node.name]}"
                node.path[-1] = node.name
        return nodes

    def _node_stage_name(self, node: KnowledgeNode) -> str:
        fallback = f"depth-{node.depth}"
        return slugify("-".join(node.path), fallback=fallback)

    def _log_child_cluster_partition_result(
        self,
        parent: KnowledgeNode,
        cluster_group: ClusterGroup,
        summary_group: list[ClusterNodeSummary],
    ) -> None:
        parts = ", ".join(
            f"{summary.name}[items={len(summary.item_ids)}]"
            for summary in summary_group
        )
        self.reporter.info(
            "Child cluster partition result: "
            f"parent={' > '.join(parent.path)} "
            f"cluster={cluster_group.cluster_id} "
            f"items={len(cluster_group.case_ids)} "
            f"nodes={len(summary_group)} "
            f"detail={parts}"
        )

    async def _resolve_case_summary_outcome(
        self,
        parent: KnowledgeNode,
        case: CaseRecord,
        outcome: TaskOutcome[str],
    ) -> str:
        if outcome.ok and outcome.value is not None:
            return outcome.value

        assert outcome.error is not None
        self.reporter.warn(
            f"Case summary failed for {case.case_id} under {' > '.join(parent.path)}, using heuristic fallback"
        )
        try:
            summary = await self.fallback_llm.summarize_case_under_parent(parent, case)
            self.audit.record_case_failure(
                stage="tree_case_summary",
                case=case,
                error=outcome.error,
                fallback="heuristic",
                fallback_succeeded=True,
                details={"parent_path": parent.path},
            )
            return summary
        except Exception as fallback_error:  # noqa: BLE001
            self.audit.record_case_failure(
                stage="tree_case_summary",
                case=case,
                error=outcome.error,
                fallback="heuristic",
                fallback_succeeded=False,
                details={
                    "parent_path": parent.path,
                    "fallback_error": str(fallback_error),
                },
            )
            return truncate(case.case_name, 40)

    async def _resolve_child_cluster_summary_outcome(
        self,
        parent: KnowledgeNode,
        cluster_group: ClusterGroup,
        cluster_cases: list[CaseRecord],
        cluster_summaries: list[str],
        outcome: TaskOutcome[list[ClusterNodeSummary]],
    ) -> list[ClusterNodeSummary]:
        if outcome.ok and outcome.value is not None:
            return outcome.value

        assert outcome.error is not None
        self.reporter.warn(
            f"Child cluster summary failed for {cluster_group.cluster_id} under {' > '.join(parent.path)}, using heuristic fallback"
        )
        try:
            summary = await self.fallback_llm.summarize_child_cluster_partitions(
                parent,
                cluster_group,
                cluster_cases,
                cluster_summaries,
            )
            self.audit.record_item_failure(
                stage="tree_child_cluster_summary",
                item_type="cluster",
                item_id=cluster_group.cluster_id,
                error=outcome.error,
                fallback="heuristic",
                fallback_succeeded=True,
                details={
                    "parent_path": parent.path,
                    "case_ids": cluster_group.case_ids,
                },
            )
            return summary
        except Exception as fallback_error:  # noqa: BLE001
            self.audit.record_item_failure(
                stage="tree_child_cluster_summary",
                item_type="cluster",
                item_id=cluster_group.cluster_id,
                error=outcome.error,
                fallback="heuristic",
                fallback_succeeded=False,
                details={
                    "parent_path": parent.path,
                    "case_ids": cluster_group.case_ids,
                    "fallback_error": str(fallback_error),
                },
            )
            name = truncate(cluster_summaries[0], 32) if cluster_summaries else cluster_group.cluster_id
            return [
                ClusterNodeSummary(
                    name=name,
                    trigger=f"当{parent.name}问题进一步表现为{name}方向时考虑该子类别",
                    background=f"{name}为回退生成的子类别，请结合原始案例进一步校验。",
                    item_ids=cluster_group.case_ids,
                )
            ]
