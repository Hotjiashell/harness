from __future__ import annotations

import json

from .models import CaseRecord, ClusterGroup, KnowledgeNode


def _json_block(payload: object) -> str:
    if isinstance(payload, str):
        content = payload.strip()
    else:
        content = json.dumps(payload, ensure_ascii=False, indent=2)
    return f"```json\n{content}\n```"


def _json_output_requirement(schema: str) -> str:
    return (
        "输出要求：\n"
        "1. 你必须返回一个且仅一个```json代码块\n"
        "2. 不要输出代码块外的任何解释、前后缀或注释\n"
        "3. 代码块中的JSON必须是合法可解析对象\n"
        "输出格式：\n"
        f"{_json_block(schema)}"
    )


def build_classification_prompt(case: CaseRecord, seed_nodes: list[KnowledgeNode]) -> list[dict[str, str]]:
    seed_payload = [
        {
            "name": node.name,
            "trigger": node.trigger,
            #"background": node.background,
        }
        for node in seed_nodes
    ]
    return [
        {
            "role": "system",
            "content": (
                "你是知识分类助手。"
                "你必须只返回一个```json代码块，且不要输出额外解释。"
                "判断案例是否属于已有L1类别。"
            ),
        },
        {
            "role": "user",
            "content": (
                "已有L1类别：\n"
                f"{_json_block(seed_payload)}\n\n"
                "案例：\n"
                f"{_json_block(case.to_dict())}\n\n"
                + _json_output_requirement(
                    '{"belongs": true, "category_name": "类别名或null", "reason": "简明理由"}'
                )
            ),
        },
    ]


def build_discovery_prompt(case: CaseRecord) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "你是知识抽取助手。"
                "你必须只返回一个```json代码块，且不要输出额外解释。"
                "如果案例与软件使用相关，提取软件名；否则软件名为空字符串。软件名一般形式为英文词语或英文字母缩写。忽略中文软件名。"
                "软件用途不是案例中软件的具体功能细节，而是应该说明软件用于什么场景，一般形式为<软件名>是用于...。"
                "对于非软件或中文软件，description应该是案例内容或软件用途的简洁概括。"
            ),
        },
        {
            "role": "user",
            "content": (
                "案例：\n"
                f"{_json_block(case.to_dict())}\n\n"
                + _json_output_requirement(
                    '{"software_name": "英文软件名或空字符串", "description": "案例内容或软件用途的简洁概括"}'
                )
            ),
        },
    ]


def build_candidate_summary_prompt(
    cluster_group: ClusterGroup,
    cases: list[CaseRecord],
) -> list[dict[str, str]]:
    cluster_payload = cluster_group.to_dict()
    return [
        {
            "role": "system",
            "content": (
                "你是知识树构建助手。"
                "你必须只返回一个```json代码块，且不要输出额外解释。"
                "根据一个候选聚类总结知识节点。你可以把一个聚类拆成1到3个节点。"
            ),
        },
        {
            "role": "user",
            "content": (
                "候选聚类：\n"
                f"{_json_block(cluster_payload)}\n\n"
                "要求：\n"
                "1. 你可以输出1到3个节点\n"
                "2. 每个节点都要给出 item_ids，item_ids 必须从 cluster.items[*].case_id 中选择\n"
                "3. 你认为应该归为同一类的 item 放到同一个节点\n"
                "4. 所有 item 都应该被分配到某个节点\n"
                "5. 如果整个聚类本质上只是一类，就只输出1个节点\n\n"
                + _json_output_requirement(
                    '{"nodes": [{"name": "类别名称", "trigger": "什么时候考虑该类别", "background": "背景知识", "item_ids": ["cluster.items[*].case_id"]}]}'
                )
            ),
        },
    ]


def build_software_name_group_summary_prompt(
    software_name: str,
    cluster_group: ClusterGroup,
) -> list[dict[str, str]]:
    payload = {
        "software_name": software_name,
        "cluster": cluster_group.to_dict(),
    }
    return [
        {
            "role": "system",
            "content": (
                "你是知识树构建助手。"
                "你必须只返回一个```json代码块，且不要输出额外解释。"
                "当前输入是同一软件名的一组聚类item，请基于这些item总结一个软件节点。"
            ),
        },
        {
            "role": "user",
            "content": (
                "同软件名组：\n"
                f"{_json_block(payload)}\n\n"
                "额外要求：\n"
                f"1. 类别名称必须显式包含软件名“{software_name}”\n"
                "2. 背景知识必须明确提到该软件的用途或主要使用场景\n"
                "3. 如果聚类item已经体现该软件的主要用途或问题方向，优先体现在名称和背景知识中\n"
                "4. 不要只输出空泛名称，例如“相关问题”或“使用问题”\n\n"
                + _json_output_requirement(
                    '{"name": "必须包含软件名的类别名称", "trigger": "什么时候考虑该类别", "background": "先说明软件用途，再说明该节点覆盖的问题范围"}'
                )
            ),
        },
    ]


def build_case_summary_prompt(parent: KnowledgeNode, case: CaseRecord) -> list[dict[str, str]]:
    payload = {
        "parent": {
            "name": parent.name,
            "trigger": parent.trigger,
            "background": parent.background,
        },
        "case": case.to_dict(),
    }
    return [
        {
            "role": "system",
            "content": (
                "你是知识树分层助手。"
                "你必须只返回一个```json代码块，且不要输出额外解释。"
                "在给定父类别上下文下，总结该案例反映了哪一种具体子问题。"
            ),
        },
        {
            "role": "user",
            "content": (
                "上下文与案例：\n"
                f"{_json_block(payload)}\n\n"
                + _json_output_requirement(
                    '{"subproblem": "该案例在当前父类别下的具体子问题"}'
                )
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
                "你是知识树构建助手。"
                "你必须只返回一个```json代码块，且不要输出额外解释。"
                "基于父节点语境为聚类结果生成名称、类别和背景知识。你可以把一个聚类拆成1到3个子节点。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"{_json_block(payload)}\n\n"
                "要求：\n"
                "1. 你可以输出1到3个子节点\n"
                "2. 每个子节点都要给出 item_ids，item_ids 必须从 cluster.items[*].case_id 中选择\n"
                "3. 你认为属于同一子类的 item 放到同一个子节点\n"
                "4. 所有 item 都应该被分配到某个子节点\n"
                "5. 如果整个聚类在当前父节点下本质上只是一类，就只输出1个子节点\n\n"
                + _json_output_requirement(
                    '{"nodes": [{"name": "类别名称", "trigger": "什么时候考虑该类别", "background": "背景知识", "item_ids": ["cluster.items[*].case_id"]}]}'
                )
            ),
        },
    ]
