from __future__ import annotations

import asyncio
import json
import re
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from config import HarnessConfig

from .models import (
    CaseRecord,
    ClassificationResult,
    ClusterNodeSummary,
    ClusterGroup,
    DiscoveryResult,
    KnowledgeNode,
    NodeSummary,
)
from .prompts import (
    build_candidate_summary_prompt,
    build_case_summary_prompt,
    build_child_summary_prompt,
    build_classification_prompt,
    build_discovery_prompt,
    build_software_name_group_summary_prompt,
)
from .utils import (
    extract_english_candidates,
    normalize_software_name,
    split_mixed_tokens,
    top_terms,
    truncate,
)


JSON_BLOCK_PATTERN = re.compile(r"```json\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)
CODE_BLOCK_PATTERN = re.compile(r"```(?:[a-zA-Z0-9_+-]+)?\s*(.*?)\s*```", re.DOTALL)
PURPOSE_KEYWORDS = ("用于", "用来", "主要用于", "常用于", "适用于")


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _extract_purpose_sentence(software_name: str, texts: list[str]) -> str:
    software_label = software_name.strip()
    for text in texts:
        cleaned = _normalize_whitespace(text)
        if not cleaned:
            continue
        has_software_name = software_label and software_label.lower() in cleaned.lower()
        if has_software_name and any(keyword in cleaned for keyword in PURPOSE_KEYWORDS):
            return cleaned if cleaned.endswith("。") else f"{cleaned}。"
        for keyword in PURPOSE_KEYWORDS:
            if keyword not in cleaned:
                continue
            fragment = cleaned.split(keyword, 1)[1]
            fragment = re.split(r"[。；;]", fragment, maxsplit=1)[0].strip("，, ：: ")
            if fragment:
                prefix = software_label or "该软件"
                return f"{prefix}通常用于{fragment}。"
    terms = top_terms(texts, limit=3)
    if software_label and terms:
        return f"{software_label}通常用于处理{'/'.join(terms)}等相关场景。"
    if software_label:
        return f"{software_label}通常用于相关业务场景。"
    return "该软件通常用于相关业务场景。"


def _background_mentions_purpose(background: str, software_name: str) -> bool:
    text = _normalize_whitespace(background)
    if not text:
        return False
    if software_name and software_name.lower() not in text.lower():
        return False
    return any(keyword in text for keyword in PURPOSE_KEYWORDS)


class LLMClient(ABC):
    @abstractmethod
    async def classify_case(
        self,
        case: CaseRecord,
        seed_nodes: list[KnowledgeNode],
    ) -> ClassificationResult:
        raise NotImplementedError

    @abstractmethod
    async def discover_case(self, case: CaseRecord) -> DiscoveryResult:
        raise NotImplementedError

    @abstractmethod
    async def summarize_candidate_cluster(
        self,
        cluster_group: ClusterGroup,
        cases: list[CaseRecord],
    ) -> NodeSummary:
        raise NotImplementedError

    @abstractmethod
    async def summarize_candidate_cluster_partitions(
        self,
        cluster_group: ClusterGroup,
        cases: list[CaseRecord],
    ) -> list[ClusterNodeSummary]:
        raise NotImplementedError

    @abstractmethod
    async def summarize_software_name_group(
        self,
        software_name: str,
        cluster_group: ClusterGroup,
        cases: list[CaseRecord],
    ) -> NodeSummary:
        raise NotImplementedError

    @abstractmethod
    async def summarize_case_under_parent(
        self,
        parent: KnowledgeNode,
        case: CaseRecord,
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    async def summarize_child_cluster(
        self,
        parent: KnowledgeNode,
        cluster_group: ClusterGroup,
        cases: list[CaseRecord],
        cluster_summaries: list[str],
    ) -> NodeSummary:
        raise NotImplementedError

    @abstractmethod
    async def summarize_child_cluster_partitions(
        self,
        parent: KnowledgeNode,
        cluster_group: ClusterGroup,
        cases: list[CaseRecord],
        cluster_summaries: list[str],
    ) -> list[ClusterNodeSummary]:
        raise NotImplementedError


@dataclass
class OpenAICompatibleLLM(LLMClient):
    config: HarnessConfig

    async def classify_case(
        self,
        case: CaseRecord,
        seed_nodes: list[KnowledgeNode],
    ) -> ClassificationResult:
        payload = await self._complete_json(build_classification_prompt(case, seed_nodes))
        belongs = bool(payload.get("belongs"))
        category_name = payload.get("category_name")
        if isinstance(category_name, str) and not category_name.strip():
            category_name = None
        return ClassificationResult(
            case_id=case.case_id,
            belongs=belongs,
            category_name=category_name,
            reason=str(payload.get("reason", "")).strip() or "模型未返回理由",
        )

    async def discover_case(self, case: CaseRecord) -> DiscoveryResult:
        payload = await self._complete_json(build_discovery_prompt(case))
        return DiscoveryResult(
            case_id=case.case_id,
            software_name=str(payload.get("software_name", "")).strip(),
            description=str(payload.get("description", "")).strip() or truncate(case.content),
        )

    async def summarize_candidate_cluster(
        self,
        cluster_group: ClusterGroup,
        cases: list[CaseRecord],
    ) -> NodeSummary:
        summaries = await self.summarize_candidate_cluster_partitions(cluster_group, cases)
        first = summaries[0]
        return NodeSummary(
            name=first.name,
            trigger=first.trigger,
            background=first.background,
        )

    async def summarize_candidate_cluster_partitions(
        self,
        cluster_group: ClusterGroup,
        cases: list[CaseRecord],
    ) -> list[ClusterNodeSummary]:
        payload = await self._complete_json(build_candidate_summary_prompt(cluster_group, cases))
        return self._normalize_cluster_node_summaries(
            payload,
            cluster_group=cluster_group,
            fallback_name="候选类别",
        )

    async def summarize_software_name_group(
        self,
        software_name: str,
        cluster_group: ClusterGroup,
        cases: list[CaseRecord],
    ) -> NodeSummary:
        payload = await self._complete_json(
            build_software_name_group_summary_prompt(software_name, cluster_group)
        )
        return self._normalize_software_name_group_summary(
            payload,
            software_name=software_name,
            cluster_group=cluster_group,
            cases=cases,
        )

    async def summarize_case_under_parent(
        self,
        parent: KnowledgeNode,
        case: CaseRecord,
    ) -> str:
        payload = await self._complete_json(build_case_summary_prompt(parent, case))
        return str(payload.get("subproblem", "")).strip() or truncate(case.case_name, 40)

    async def summarize_child_cluster(
        self,
        parent: KnowledgeNode,
        cluster_group: ClusterGroup,
        cases: list[CaseRecord],
        cluster_summaries: list[str],
    ) -> NodeSummary:
        summaries = await self.summarize_child_cluster_partitions(
            parent,
            cluster_group,
            cases,
            cluster_summaries,
        )
        first = summaries[0]
        return NodeSummary(
            name=first.name,
            trigger=first.trigger,
            background=first.background,
        )

    async def summarize_child_cluster_partitions(
        self,
        parent: KnowledgeNode,
        cluster_group: ClusterGroup,
        cases: list[CaseRecord],
        cluster_summaries: list[str],
    ) -> list[ClusterNodeSummary]:
        payload = await self._complete_json(
            build_child_summary_prompt(parent, cluster_group, cases, cluster_summaries)
        )
        return self._normalize_cluster_node_summaries(
            payload,
            cluster_group=cluster_group,
            fallback_name="子类别",
        )

    def _normalize_summary(self, payload: dict[str, Any], fallback_name: str) -> NodeSummary:
        name = str(payload.get("name", "")).strip() or fallback_name
        trigger = str(payload.get("trigger", "")).strip() or f"当问题表现为{name}相关场景时考虑该类别"
        background = str(payload.get("background", "")).strip() or f"{name}相关问题需要结合案例和父类上下文分析。"
        return NodeSummary(name=name, trigger=trigger, background=background)

    def _normalize_software_name_group_summary(
        self,
        payload: dict[str, Any],
        software_name: str,
        cluster_group: ClusterGroup,
        cases: list[CaseRecord],
    ) -> NodeSummary:
        software_label = software_name.strip() or "该软件"
        summary = self._normalize_summary(payload, fallback_name=f"{software_label}相关问题")
        name = summary.name
        if software_label.lower() not in name.lower():
            name = f"{software_label} {name}".strip()

        trigger = summary.trigger
        if software_label.lower() not in trigger.lower():
            trigger = f"当问题涉及{software_label}使用，尤其是{name}相关场景时考虑该类别"

        purpose_sentence = _extract_purpose_sentence(
            software_label,
            [
                item.description
                for item in cluster_group.items
                if item.description.strip()
            ]
            + [
                item.text
                for item in cluster_group.items
                if item.text.strip()
            ]
            + [case.case_name for case in cases],
        )
        background = summary.background
        if not _background_mentions_purpose(background, software_label):
            background = f"{purpose_sentence} {background}".strip()

        return NodeSummary(name=name, trigger=trigger, background=background)

    def _normalize_cluster_node_summaries(
        self,
        payload: dict[str, Any],
        cluster_group: ClusterGroup,
        fallback_name: str,
    ) -> list[ClusterNodeSummary]:
        valid_item_ids = list(dict.fromkeys(cluster_group.case_ids))
        if not valid_item_ids:
            return []

        raw_nodes = payload.get("nodes")
        if not isinstance(raw_nodes, list):
            raw_nodes = [payload]
        raw_nodes = [node for node in raw_nodes if isinstance(node, dict)]
        if not raw_nodes:
            raw_nodes = [{}]

        if len(raw_nodes) > 3:
            merged_nodes = list(raw_nodes[:3])
            overflow_item_ids: list[str] = []
            for extra_node in raw_nodes[3:]:
                raw_item_ids = extra_node.get("item_ids")
                if isinstance(raw_item_ids, list):
                    overflow_item_ids.extend(str(item_id) for item_id in raw_item_ids)
            merged_last_item_ids = merged_nodes[-1].get("item_ids")
            if isinstance(merged_last_item_ids, list):
                merged_last_item_ids.extend(overflow_item_ids)
            else:
                merged_nodes[-1]["item_ids"] = overflow_item_ids
            raw_nodes = merged_nodes

        remaining_ids = list(valid_item_ids)
        remaining_set = set(remaining_ids)
        results: list[ClusterNodeSummary] = []

        for index, node_payload in enumerate(raw_nodes, start=1):
            raw_item_ids = node_payload.get("item_ids")
            normalized_item_ids: list[str] = []
            if isinstance(raw_item_ids, list):
                seen_item_ids: set[str] = set()
                for raw_item_id in raw_item_ids:
                    item_id = str(raw_item_id)
                    if item_id not in remaining_set or item_id in seen_item_ids:
                        continue
                    normalized_item_ids.append(item_id)
                    seen_item_ids.add(item_id)

            if not normalized_item_ids:
                continue

            remaining_set.difference_update(normalized_item_ids)
            remaining_ids = [item_id for item_id in remaining_ids if item_id in remaining_set]

            summary = self._normalize_summary(
                node_payload,
                fallback_name=f"{fallback_name}{index}" if len(raw_nodes) > 1 else fallback_name,
            )
            results.append(
                ClusterNodeSummary(
                    name=summary.name,
                    trigger=summary.trigger,
                    background=summary.background,
                    item_ids=normalized_item_ids,
                )
            )

        if not results:
            summary = self._normalize_summary(payload, fallback_name=fallback_name)
            return [
                ClusterNodeSummary(
                    name=summary.name,
                    trigger=summary.trigger,
                    background=summary.background,
                    item_ids=valid_item_ids,
                )
            ]

        if remaining_ids:
            results[0].item_ids.extend(remaining_ids)

        return results

    async def _complete_json(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        last_error: Exception | None = None
        for _ in range(self.config.llm.max_retries + 1):
            try:
                return await self._call_chat_completion(messages)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                await asyncio.sleep(0.5)
        assert last_error is not None
        raise last_error

    async def _call_chat_completion(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        endpoint = self.config.llm.base_url.rstrip("/")
        if not endpoint.endswith("/chat/completions"):
            endpoint = f"{endpoint}/chat/completions"

        request_payload = {
            "model": self.config.llm.model,
            "messages": messages,
            "temperature": self.config.llm.temperature,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.llm.api_key}",
        }

        def do_request() -> dict[str, Any]:
            request = Request(
                endpoint,
                data=json.dumps(request_payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            try:
                with urlopen(request, timeout=self.config.llm.timeout_seconds) as response:
                    raw = response.read().decode("utf-8")
            except HTTPError as exc:
                body = exc.read().decode("utf-8", errors="ignore")
                raise RuntimeError(f"LLM request failed: {exc.code} {body}") from exc
            except URLError as exc:
                raise RuntimeError(f"LLM request failed: {exc}") from exc

            response_payload = json.loads(raw)
            choices = response_payload.get("choices") or []
            if not choices:
                raise RuntimeError(f"LLM response missing choices: {response_payload}")
            content = (
                choices[0]
                .get("message", {})
                .get("content", "")
            )
            return parse_json_payload(content)

        return await asyncio.to_thread(do_request)


@dataclass
class HeuristicLLM(LLMClient):
    config: HarnessConfig

    async def classify_case(
        self,
        case: CaseRecord,
        seed_nodes: list[KnowledgeNode],
    ) -> ClassificationResult:
        content = case.content.lower()
        best_node: KnowledgeNode | None = None
        best_score = 0
        for node in seed_nodes:
            score = 0
            node_name = node.name.lower()
            if node_name in content:
                score += 10
            for token in split_mixed_tokens(f"{node.trigger} {node.background}"):
                if len(token) <= 1:
                    continue
                if token.lower() in content:
                    score += 1
            if score > best_score:
                best_score = score
                best_node = node

        if best_node and best_score >= 3:
            return ClassificationResult(
                case_id=case.case_id,
                belongs=True,
                category_name=best_node.name,
                reason=f"命中种子类别关键词，匹配分数为 {best_score}",
            )

        return ClassificationResult(
            case_id=case.case_id,
            belongs=False,
            category_name=None,
            reason="未命中任何已有L1类别的显著关键词",
        )

    async def discover_case(self, case: CaseRecord) -> DiscoveryResult:
        candidates = extract_english_candidates(case.case_name)
        if not candidates:
            candidates = [
                token
                for token in extract_english_candidates(case.text)
                if self._looks_like_software_name(token)
            ]
        software_name = candidates[0] if candidates else ""

        description = case.case_name
        if software_name:
            description = re.sub(
                re.escape(software_name),
                "",
                description,
                flags=re.IGNORECASE,
            ).strip(" -_：:（）()")
        if not description:
            description = truncate(case.text, 60)
        return DiscoveryResult(
            case_id=case.case_id,
            software_name=software_name,
            description=description,
        )

    async def summarize_candidate_cluster(
        self,
        cluster_group: ClusterGroup,
        cases: list[CaseRecord],
    ) -> NodeSummary:
        name = self._infer_name(cluster_group, cases)
        return NodeSummary(
            name=name,
            trigger=f"当问题表现为{name}相关场景，或案例语义聚类与该组相近时考虑该类别",
            background=self._build_background(name, cases),
        )

    async def summarize_candidate_cluster_partitions(
        self,
        cluster_group: ClusterGroup,
        cases: list[CaseRecord],
    ) -> list[ClusterNodeSummary]:
        summary = await self.summarize_candidate_cluster(cluster_group, cases)
        return [
            ClusterNodeSummary(
                name=summary.name,
                trigger=summary.trigger,
                background=summary.background,
                item_ids=cluster_group.case_ids,
            )
        ]

    async def summarize_software_name_group(
        self,
        software_name: str,
        cluster_group: ClusterGroup,
        cases: list[CaseRecord],
    ) -> NodeSummary:
        software_label = software_name.strip() or self._infer_name(cluster_group, cases)
        focus_terms = top_terms(
            [
                re.sub(re.escape(software_label), "", case.case_name, flags=re.IGNORECASE).strip(" -_：:（）()")
                for case in cases
            ],
            limit=2,
        )
        focus = "/".join(term for term in focus_terms if term) if focus_terms else ""
        name = f"{software_label} {focus}".strip() if focus else f"{software_label}相关问题"
        purpose_sentence = _extract_purpose_sentence(
            software_label,
            [
                item.description
                for item in cluster_group.items
                if item.description.strip()
            ]
            + [
                item.text
                for item in cluster_group.items
                if item.text.strip()
            ]
            + [case.case_name for case in cases],
        )
        return NodeSummary(
            name=name,
            trigger=f"当问题涉及{software_label}使用，尤其是{name}相关场景时考虑该类别",
            background=f"{purpose_sentence} 该节点汇总了{name}相关的同软件案例。",
        )

    async def summarize_case_under_parent(
        self,
        parent: KnowledgeNode,
        case: CaseRecord,
    ) -> str:
        summary = case.case_name
        if parent.name.lower() in summary.lower():
            summary = re.sub(parent.name, "", summary, flags=re.IGNORECASE).strip(" -_：:（）()")
        return summary or truncate(case.text, 40)

    async def summarize_child_cluster(
        self,
        parent: KnowledgeNode,
        cluster_group: ClusterGroup,
        cases: list[CaseRecord],
        cluster_summaries: list[str],
    ) -> NodeSummary:
        if len(cluster_summaries) == 1:
            name = truncate(cluster_summaries[0], 32)
        else:
            terms = top_terms(cluster_summaries or [case.case_name for case in cases], limit=2)
            if terms:
                name = "/".join(terms)
            else:
                name = self._infer_name(cluster_group, cases)
        return NodeSummary(
            name=name,
            trigger=f"当{parent.name}问题进一步表现为{name}方向时考虑该子类别",
            background=(
                f"{name}属于{parent.name}下更具体的问题簇，"
                "可结合该簇案例中的共性处理动作进行判断。"
            ),
        )

    async def summarize_child_cluster_partitions(
        self,
        parent: KnowledgeNode,
        cluster_group: ClusterGroup,
        cases: list[CaseRecord],
        cluster_summaries: list[str],
    ) -> list[ClusterNodeSummary]:
        summary = await self.summarize_child_cluster(
            parent,
            cluster_group,
            cases,
            cluster_summaries,
        )
        return [
            ClusterNodeSummary(
                name=summary.name,
                trigger=summary.trigger,
                background=summary.background,
                item_ids=cluster_group.case_ids,
            )
        ]

    def _infer_name(self, cluster_group: ClusterGroup, cases: list[CaseRecord]) -> str:
        software_names = [
            item.software_name.strip()
            for item in cluster_group.items
            if item.software_name.strip()
        ]
        if len(cases) == 1 and not software_names:
            return truncate(cases[0].case_name, 28)
        if software_names:
            counter = Counter(normalize_software_name(name) for name in software_names if name)
            normalized = counter.most_common(1)[0][0]
            for original in software_names:
                if normalize_software_name(original) == normalized:
                    return original

        terms = top_terms([case.case_name for case in cases], limit=2)
        if terms:
            return "/".join(terms)
        return truncate(cases[0].case_name, 20) if cases else "未命名类别"

    def _build_background(self, name: str, cases: list[CaseRecord]) -> str:
        examples = "；".join(truncate(case.case_name, 28) for case in cases[:3])
        return f"{name}相关知识由相似案例归纳而来，典型示例包括：{examples}。"

    def _looks_like_software_name(self, token: str) -> bool:
        if token.isupper():
            return True
        return any(char.isupper() for char in token[1:])


def parse_json_payload(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if not stripped:
        raise ValueError("LLM returned empty content")

    candidates: list[str] = []
    json_blocks = [match.group(1).strip() for match in JSON_BLOCK_PATTERN.finditer(stripped)]
    if json_blocks:
        candidates.extend(json_blocks)
    else:
        generic_blocks = [match.group(1).strip() for match in CODE_BLOCK_PATTERN.finditer(stripped)]
        candidates.extend(generic_blocks)
    candidates.append(stripped)

    last_error: Exception | None = None
    for candidate in candidates:
        if not candidate:
            continue
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            start = candidate.find("{")
            end = candidate.rfind("}")
            if start == -1 or end == -1 or end <= start:
                continue
            try:
                payload = json.loads(candidate[start : end + 1])
            except json.JSONDecodeError as inner_exc:
                last_error = inner_exc
                continue
        if not isinstance(payload, dict):
            raise ValueError(f"Expected JSON object, got: {payload}")
        return payload

    if last_error is not None:
        raise last_error
    raise ValueError("Failed to parse JSON payload from LLM response")


def create_llm_client(config: HarnessConfig) -> LLMClient:
    provider = config.llm.provider.strip().lower()
    if provider == "openai-compatible":
        return OpenAICompatibleLLM(config=config)
    if provider == "heuristic":
        return HeuristicLLM(config=config)
    raise ValueError(f"Unsupported llm provider: {config.llm.provider}")
