from __future__ import annotations

import json

from .models import CaseRecord, ClusterGroup, KnowledgeNode


def build_classification_prompt(case: CaseRecord, seed_nodes: list[KnowledgeNode]) -> list[dict[str, str]]:
    seed_payload = [
        {
            "name": node.name,
            "trigger": node.trigger,
            "background": node.background,
        }
        for node in seed_nodes
    ]
    return [
        {
            "role": "system",
            "content": (
                "你是知识分类助手。请只输出JSON，不要输出额外解释。"
                "判断案例是否属于已有L1类别。"
            ),
        },
        {
            "role": "user",
            "content": (
                "已有L1类别：\n"
                f"{json.dumps(seed_payload, ensure_ascii=False, indent=2)}\n\n"
                "案例：\n"
                f"{json.dumps(case.to_dict(), ensure_ascii=False, indent=2)}\n\n"
                "输出格式："
                '{"belongs": true/false, "category_name": "类别名或null", "reason": "简明理由"}'
            ),
        },
    ]


def build_discovery_prompt(case: CaseRecord) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "你是知识抽取助手。请只输出JSON，不要输出额外解释。"
                "如果案例与软件使用相关，提取软件名；否则软件名为空字符串。"
            ),
        },
        {
            "role": "user",
            "content": (
                "案例：\n"
                f"{json.dumps(case.to_dict(), ensure_ascii=False, indent=2)}\n\n"
                "输出格式："
                '{"software_name": "软件名或空字符串", "description": "案例内容或软件功能的简洁概括"}'
            ),
        },
    ]


def build_candidate_summary_prompt(
    cluster_group: ClusterGroup,
    cases: list[CaseRecord],
) -> list[dict[str, str]]:
    case_payload = [case.to_dict() for case in cases]
    cluster_payload = cluster_group.to_dict()
    return [
        {
            "role": "system",
            "content": (
                "你是知识树构建助手。请只输出JSON，不要输出额外解释。"
                "根据一个候选聚类总结知识节点。"
            ),
        },
        {
            "role": "user",
            "content": (
                "候选聚类：\n"
                f"{json.dumps(cluster_payload, ensure_ascii=False, indent=2)}\n\n"
                "相关案例：\n"
                f"{json.dumps(case_payload, ensure_ascii=False, indent=2)}\n\n"
                "输出格式："
                '{"name": "类别名称", "trigger": "什么时候考虑该类别", "background": "背景知识"}'
            ),
        },
    ]


def build_case_summary_prompt(parent: KnowledgeNode, case: CaseRecord) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "你是知识树分层助手。请只输出JSON，不要输出额外解释。"
                "在给定父类别上下文下，总结该案例属于哪一种具体子问题。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Parent Name: {parent.name}\n"
                f"Parent Background Knowledge: {parent.background}\n"
                f"Case Content: {case.content}\n\n"
                '输出格式：{"subproblem": "该案例在当前父类别下的具体子问题"}'
            ),
        },
    ]


def build_child_summary_prompt(
    parent: KnowledgeNode,
    cluster_group: ClusterGroup,
    cases: list[CaseRecord],
    cluster_summaries: list[str],
) -> list[dict[str, str]]:
    payload = {
        "parent": {
            "name": parent.name,
            "trigger": parent.trigger,
            "background": parent.background,
        },
        "cluster": cluster_group.to_dict(),
        "cluster_summaries": cluster_summaries,
        "cases": [case.to_dict() for case in cases],
    }
    return [
        {
            "role": "system",
            "content": (
                "你是知识树构建助手。请只输出JSON，不要输出额外解释。"
                "基于父节点语境和聚类结果生成更具体的子节点。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
                '输出格式：{"name": "类别名称", "trigger": "什么时候考虑该类别", "background": "背景知识"}'
            ),
        },
    ]
