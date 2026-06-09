from __future__ import annotations

from dataclasses import asdict

from config import CONFIG, HarnessConfig

from .exporter import export_skill_tree
from .io_utils import load_cases, load_seed_l1, write_json, write_tree_outputs
from .l1 import L1Builder
from .llm import create_llm_client
from .tree import RecursiveTreeBuilder
from .utils import ensure_directory


async def build_harness(config: HarnessConfig = CONFIG):
    ensure_directory(config.paths.output_dir)
    stage_dir = ensure_directory(config.paths.output_dir / config.pipeline.stage_dir_name)
    write_json(
        config.paths.output_dir / config.pipeline.config_snapshot_filename,
        asdict(config),
    )

    cases = load_cases(config.paths.case_path)
    seed_nodes = load_seed_l1(config.paths.seed_l1_path)
    cases_by_id = {case.case_id: case for case in cases}
    llm_client = create_llm_client(config)

    l1_builder = L1Builder(config, llm_client, stage_dir)
    root = await l1_builder.build(cases, seed_nodes)

    tree_builder = RecursiveTreeBuilder(config, llm_client, stage_dir, cases_by_id)
    await tree_builder.build(root)

    write_tree_outputs(
        root,
        config.paths.output_dir / config.pipeline.final_tree_filename,
        config.paths.output_dir / config.pipeline.debug_tree_filename,
    )
    export_skill_tree(root, cases_by_id, config.paths.skills_dir)
    return root
