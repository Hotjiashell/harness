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
)
from .utils import (
    extract_english_candidates,
    normalize_software_name,
    split_mixed_tokens,
    top_terms,
    truncate,
)


JSON_BLOCK_PATTERN = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL)


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
        payload = await self._complete_json(build_candidate_summary_prompt(cluster_group, cases))
        return self._normalize_summary(payload, fallback_name="候选类别")

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
        payload = await self._complete_json(
            build_child_summary_prompt(parent, cluster_group, cases, cluster_summaries)
        )
        return self._normalize_summary(payload, fallback_name="子类别")

    def _normalize_summary(self, payload: dict[str, Any], fallback_name: str) -> NodeSummary:
        name = str(payload.get("name", "")).strip() or fallback_name
        trigger = str(payload.get("trigger", "")).strip() or f"当问题表现为{name}相关场景时考虑该类别"
        background = str(payload.get("background", "")).strip() or f"{name}相关问题需要结合案例和父类上下文分析。"
        return NodeSummary(name=name, trigger=trigger, background=background)

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

        payload = {
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
                data=json.dumps(payload).encode("utf-8"),
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

            payload = json.loads(raw)
            choices = payload.get("choices") or []
            if not choices:
                raise RuntimeError(f"LLM response missing choices: {payload}")
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

    block_match = JSON_BLOCK_PATTERN.search(stripped)
    if block_match:
        stripped = block_match.group(1).strip()

    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        payload = json.loads(stripped[start : end + 1])

    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object, got: {payload}")
    return payload


def create_llm_client(config: HarnessConfig) -> LLMClient:
    provider = config.llm.provider.strip().lower()
    if provider == "openai-compatible":
        return OpenAICompatibleLLM(config=config)
    if provider == "heuristic":
        return HeuristicLLM(config=config)
    raise ValueError(f"Unsupported llm provider: {config.llm.provider}")
