"""Tests for the core pipeline engine."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.pipeline import Pipeline, PipelineResult
from src.steps import PipelineStep, StepResult

PIPELINES_DIR = Path(__file__).parent.parent / "pipelines"


class TestPipelineResult:
    """Tests for PipelineResult."""

    def test_default_values(self) -> None:
        result = PipelineResult()
        assert result.success is True
        assert result.total_latency_ms == 0.0
        assert result.total_tokens == 0
        assert result.steps_results == []

    def test_to_dict(self) -> None:
        result = PipelineResult(
            total_latency_ms=500.5,
            total_tokens=100,
            total_cost=0.005,
            success=True,
            final_output="done",
        )
        d = result.to_dict()
        assert d["total_latency_ms"] == 500.5
        assert d["total_tokens"] == 100
        assert d["success"] is True

    def test_failed_steps_property(self) -> None:
        result = PipelineResult(
            steps_results=[
                StepResult(success=True, step_name="a"),
                StepResult(success=False, step_name="b", error="fail"),
                StepResult(success=True, step_name="c", skipped=True),
            ]
        )
        assert len(result.failed_steps) == 1
        assert result.failed_steps[0].step_name == "b"

    def test_successful_steps_property(self) -> None:
        result = PipelineResult(
            steps_results=[
                StepResult(success=True, step_name="a"),
                StepResult(success=False, step_name="b"),
            ]
        )
        assert len(result.successful_steps) == 1

    def test_skipped_steps_property(self) -> None:
        result = PipelineResult(
            steps_results=[
                StepResult(success=True, step_name="a"),
                StepResult(success=True, step_name="b", skipped=True),
            ]
        )
        assert len(result.skipped_steps) == 1
        assert result.skipped_steps[0].step_name == "b"


class TestPipeline:
    """Tests for Pipeline."""

    def test_create_pipeline(self) -> None:
        pipeline = Pipeline(name="test", description="A test pipeline")
        assert pipeline.name == "test"
        assert pipeline.description == "A test pipeline"
        assert pipeline.version == "1.0"
        assert len(pipeline.steps) == 0

    def test_add_step(self) -> None:
        pipeline = Pipeline(name="test")
        step = PipelineStep(name="step1", model="gpt-4", prompt_template="hello")
        result = pipeline.add_step(step)
        assert result is pipeline  # Chaining support
        assert len(pipeline) == 1

    def test_validate_empty_pipeline(self) -> None:
        pipeline = Pipeline(name="test")
        errors = pipeline.validate()
        assert any("at least one step" in e for e in errors)

    def test_validate_no_name(self) -> None:
        pipeline = Pipeline(name="")
        errors = pipeline.validate()
        assert any("name is required" in e for e in errors)

    def test_validate_duplicate_step_names(self) -> None:
        pipeline = Pipeline(
            name="test",
            steps=[
                PipelineStep(name="step1", model="gpt-4", prompt_template="a"),
                PipelineStep(name="step1", model="gpt-4", prompt_template="b"),
            ],
        )
        errors = pipeline.validate()
        assert any("Duplicate step name" in e for e in errors)

    def test_validate_unknown_depends_on(self) -> None:
        pipeline = Pipeline(
            name="test",
            steps=[
                PipelineStep(
                    name="step1",
                    model="gpt-4",
                    prompt_template="a",
                    depends_on="nonexistent",
                ),
            ],
        )
        errors = pipeline.validate()
        assert any("unknown step" in e for e in errors)

    def test_validate_valid_pipeline(self) -> None:
        pipeline = Pipeline(
            name="test",
            steps=[
                PipelineStep(name="step1", model="gpt-4", prompt_template="hello"),
                PipelineStep(
                    name="step2",
                    model="gpt-4",
                    prompt_template="world",
                    depends_on="step1",
                ),
            ],
        )
        errors = pipeline.validate()
        assert errors == []

    def test_to_dict(self) -> None:
        pipeline = Pipeline(
            name="test",
            version="2.0",
            description="desc",
            steps=[PipelineStep(name="s1", model="gpt-4", prompt_template="p")],
        )
        d = pipeline.to_dict()
        assert d["name"] == "test"
        assert d["version"] == "2.0"
        assert len(d["steps"]) == 1

    def test_get_step(self) -> None:
        pipeline = Pipeline(
            name="test",
            steps=[PipelineStep(name="find_me", model="gpt-4", prompt_template="p")],
        )
        assert pipeline.get_step("find_me") is not None
        assert pipeline.get_step("not_found") is None

    def test_repr(self) -> None:
        pipeline = Pipeline(name="my_pipe", version="1.5")
        r = repr(pipeline)
        assert "my_pipe" in r
        assert "1.5" in r


class TestPipelineYAML:
    """Tests for YAML serialization/deserialization."""

    def test_to_yaml_and_back(self) -> None:
        pipeline = Pipeline(
            name="roundtrip",
            version="1.0",
            description="Test roundtrip",
            steps=[
                PipelineStep(
                    name="step1",
                    model="gpt-4",
                    prompt_template="Hello {name}",
                    output_key="greeting",
                ),
            ],
        )
        yaml_str = pipeline.to_yaml()
        loaded = Pipeline.from_yaml(yaml_str)
        assert loaded.name == "roundtrip"
        assert len(loaded.steps) == 1
        assert loaded.steps[0].name == "step1"
        assert loaded.steps[0].model == "gpt-4"

    def test_from_yaml_invalid(self) -> None:
        with pytest.raises(ValueError, match="Invalid YAML"):
            Pipeline.from_yaml("just a string")

    def test_from_yaml_file(self) -> None:
        yaml_file = PIPELINES_DIR / "research_and_summarize.yaml"
        if yaml_file.exists():
            pipeline = Pipeline.from_yaml_file(yaml_file)
            assert pipeline.name == "research_and_summarize"
            assert len(pipeline.steps) == 3

    def test_from_yaml_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            Pipeline.from_yaml_file("/nonexistent/path.yaml")

    def test_load_all_pipeline_files(self) -> None:
        """Verify all YAML pipeline files load and validate."""
        yaml_files = list(PIPELINES_DIR.glob("*.yaml"))
        assert len(yaml_files) >= 4, "Expected at least 4 pipeline YAML files"

        for yaml_file in yaml_files:
            pipeline = Pipeline.from_yaml_file(yaml_file)
            errors = pipeline.validate()
            assert errors == [], f"Pipeline {yaml_file.name} has errors: {errors}"
