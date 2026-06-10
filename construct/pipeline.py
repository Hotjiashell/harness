from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from config import CONFIG, HarnessConfig

from .audit import ErrorAuditCollector
from .exporter import export_skill_tree
from .io_utils import load_cases, load_knowledge_node, load_seed_l1, write_json, write_tree_outputs
from .l1 import L1Builder
from .llm import create_llm_client
from .reporting import ConsoleReporter
from .tree import RecursiveTreeBuilder
from .utils import ensure_directory


def _resolve_resume_tree_path(config: HarnessConfig) -> Path | None:
    resume_tree_path = config.pipeline.resume_tree_path
    if resume_tree_path is None:
        return None
    return Path(resume_tree_path).expanduser()


def _pipeline_mode(config: HarnessConfig) -> str:
    if _resolve_resume_tree_path(config) is not None:
        return "resume"
    if config.pipeline.stop_after_l1:
        return "l1-only"
    return "full"


def _validate_pipeline_controls(config: HarnessConfig) -> None:
    resume_tree_path = _resolve_resume_tree_path(config)
    if config.pipeline.stop_after_l1 and resume_tree_path is not None:
        raise ValueError("pipeline.stop_after_l1 and pipeline.resume_tree_path cannot both be set")


def _export_current_tree(
    root,
    cases_by_id,
    config: HarnessConfig,
    reporter: ConsoleReporter,
    mode_label: str,
) -> None:
    reporter.section("Export Outputs", mode_label)
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


async def build_harness(config: HarnessConfig = CONFIG):
    _validate_pipeline_controls(config)
    reporter = ConsoleReporter(
        enabled=config.pipeline.console_output,
        log_timestamps=config.pipeline.log_timestamps,
        progress_bar_width=config.pipeline.progress_bar_width,
    )
    audit = ErrorAuditCollector()
    ensure_directory(config.paths.output_dir)
    stage_dir = ensure_directory(config.paths.output_dir / config.pipeline.stage_dir_name)
    run_mode = _pipeline_mode(config)
    reporter.section(
        "Harness Build Start",
        f"mode={run_mode} provider={config.llm.provider} output={config.paths.output_dir}",
    )
    write_json(
        config.paths.output_dir / config.pipeline.config_snapshot_filename,
        asdict(config),
    )

    cases = load_cases(config.paths.case_path)
    cases_by_id = {case.case_id: case for case in cases}
    llm_client = create_llm_client(config)
    reporter.info(f"Loaded {len(cases)} cases")

    root = None
    try:
        resume_tree_path = _resolve_resume_tree_path(config)
        if resume_tree_path is not None:
            reporter.section("Resume Tree", f"path={resume_tree_path}")
            root = load_knowledge_node(resume_tree_path, require_case_ids=True)
            reporter.info(
                f"Loaded resume tree root={root.name} depth={root.depth} children={len(root.children)}"
            )
        else:
            seed_nodes = load_seed_l1(config.paths.seed_l1_path)
            reporter.info(f"Loaded {len(seed_nodes)} seed L1 nodes")
            l1_builder = L1Builder(config, llm_client, stage_dir, reporter, audit)
            root = await l1_builder.build(cases, seed_nodes)

            if config.pipeline.stop_after_l1:
                reporter.section("Stage Control", "stop_after_l1=True, skipping L2/L3 build")
                _export_current_tree(
                    root=root,
                    cases_by_id=cases_by_id,
                    config=config,
                    reporter=reporter,
                    mode_label="L1 snapshot only",
                )
                reporter.section("Harness Build Complete", "stopped after L1")
                return root

        tree_builder = RecursiveTreeBuilder(config, llm_client, stage_dir, cases_by_id, reporter, audit)
        await tree_builder.build(root)

        _export_current_tree(
            root=root,
            cases_by_id=cases_by_id,
            config=config,
            reporter=reporter,
            mode_label="Full tree",
        )
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
