from __future__ import annotations

from dataclasses import asdict

from config import CONFIG, HarnessConfig

from .audit import ErrorAuditCollector
from .exporter import export_skill_tree
from .io_utils import load_cases, load_seed_l1, write_json, write_tree_outputs
from .l1 import L1Builder
from .llm import create_llm_client
from .reporting import ConsoleReporter
from .tree import RecursiveTreeBuilder
from .utils import ensure_directory


async def build_harness(config: HarnessConfig = CONFIG):
    reporter = ConsoleReporter(
        enabled=config.pipeline.console_output,
        log_timestamps=config.pipeline.log_timestamps,
        progress_bar_width=config.pipeline.progress_bar_width,
    )
    audit = ErrorAuditCollector()
    ensure_directory(config.paths.output_dir)
    stage_dir = ensure_directory(config.paths.output_dir / config.pipeline.stage_dir_name)
    reporter.section(
        "Harness Build Start",
        f"provider={config.llm.provider} output={config.paths.output_dir}",
    )
    write_json(
        config.paths.output_dir / config.pipeline.config_snapshot_filename,
        asdict(config),
    )

    cases = load_cases(config.paths.case_path)
    seed_nodes = load_seed_l1(config.paths.seed_l1_path)
    cases_by_id = {case.case_id: case for case in cases}
    llm_client = create_llm_client(config)
    reporter.info(f"Loaded {len(cases)} cases and {len(seed_nodes)} seed L1 nodes")

    root = None
    try:
        l1_builder = L1Builder(config, llm_client, stage_dir, reporter, audit)
        root = await l1_builder.build(cases, seed_nodes)

        tree_builder = RecursiveTreeBuilder(config, llm_client, stage_dir, cases_by_id, reporter, audit)
        await tree_builder.build(root)

        reporter.section("Export Outputs")
        write_tree_outputs(
            root,
            config.paths.output_dir / config.pipeline.final_tree_filename,
            config.paths.output_dir / config.pipeline.debug_tree_filename,
        )
        export_skill_tree(root, cases_by_id, config.paths.skills_dir)
        reporter.info(
            f"Knowledge tree written to {config.paths.output_dir / config.pipeline.final_tree_filename}"
        )
        reporter.info(f"Skill-style directory written to {config.paths.skills_dir}")
        reporter.section("Harness Build Complete")
        return root
    except Exception as exc:  # noqa: BLE001
        audit.record_item_failure(
            stage="pipeline",
            item_type="system",
            item_id="build_harness",
            error=exc,
            details={"provider": config.llm.provider},
        )
        reporter.warn("Unhandled pipeline error, audit files will still be written")
        raise
    finally:
        reporter.section("Error Audit")
        write_json(config.paths.output_dir / "error_audit.json", audit.to_dict())
        write_json(config.paths.output_dir / "failed_cases.json", audit.failed_cases_payload())
        reporter.info(
            f"Recorded {audit.summary()['total_errors']} errors across {audit.summary()['failed_case_count']} cases"
        )
