from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from config import HarnessConfig

from .clustering import cluster_items
from .io_utils import write_json
from .llm import LLMClient
from .models import CaseRecord, ClusterGroup, ClusterItem, KnowledgeNode
from .reporting import ConsoleReporter
from .utils import bounded_gather, slugify


class RecursiveTreeBuilder:
    def __init__(
        self,
        config: HarnessConfig,
        llm_client: LLMClient,
        stage_dir: Path,
        cases_by_id: dict[str, CaseRecord],
        reporter: ConsoleReporter,
    ) -> None:
        self.config = config
        self.llm = llm_client
        self.stage_dir = stage_dir
        self.cases_by_id = cases_by_id
        self.reporter = reporter

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
        return await bounded_gather(
            factories,
            self.config.llm.concurrency,
            reporter=self.reporter,
            progress_label=f"Summarize cases for {parent.name}",
        )

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
                self.llm.summarize_child_cluster(
                    parent,
                    cluster_group,
                    cluster_cases,
                    cluster_summaries,
                )
            )
            for cluster_group, cluster_cases, cluster_summaries in cluster_payloads
        ]
        summaries = await bounded_gather(
            factories,
            self.config.llm.concurrency,
            reporter=self.reporter,
            progress_label=f"Summarize child clusters for {parent.name}",
        )

        results: list[KnowledgeNode] = []
        for (cluster_group, _, _), summary in zip(cluster_payloads, summaries):
            child = KnowledgeNode(
                name=summary.name,
                trigger=summary.trigger,
                background=summary.background,
                case_ids=cluster_group.case_ids,
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
