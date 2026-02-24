"""Tests for pipeline step definitions."""

from __future__ import annotations

import pytest

from src.steps import (
    PipelineStep,
    StepResult,
    StepType,
    create_conditional_step,
    create_llm_step,
    create_parallel_group,
    create_transform_step,
)


class TestStepResult:
    """Tests for StepResult."""

    def test_default_values(self) -> None:
        result = StepResult()
        assert result.success is True
        assert result.output is None
        assert result.latency_ms == 0.0
        assert result.tokens_used == 0
        assert result.error is None
        assert result.skipped is False

    def test_to_dict(self) -> None:
        result = StepResult(
            output="test output",
            latency_ms=123.456,
            tokens_used=50,
            model_used="gpt-4",
            success=True,
            step_name="step1",
        )
        d = result.to_dict()
        assert d["output"] == "test output"
        assert d["latency_ms"] == 123.46
        assert d["tokens_used"] == 50
        assert d["model_used"] == "gpt-4"
        assert d["success"] is True
        assert d["step_name"] == "step1"

    def test_failed_result(self) -> None:
        result = StepResult(success=False, error="Model timeout")
        assert result.success is False
        assert result.error == "Model timeout"


class TestPipelineStep:
    """Tests for PipelineStep."""

    def test_default_output_key(self) -> None:
        step = PipelineStep(name="analyze")
        assert step.output_key == "analyze_output"

    def test_explicit_output_key(self) -> None:
        step = PipelineStep(name="analyze", output_key="my_key")
        assert step.output_key == "my_key"

    def test_validate_llm_step_valid(self) -> None:
        step = PipelineStep(
            name="test",
            model="gpt-4",
            prompt_template="Hello {name}",
            step_type=StepType.LLM,
        )
        errors = step.validate()
        assert errors == []

    def test_validate_llm_step_missing_model(self) -> None:
        step = PipelineStep(
            name="test",
            prompt_template="Hello",
            step_type=StepType.LLM,
        )
        errors = step.validate()
        assert any("model is required" in e for e in errors)

    def test_validate_llm_step_missing_prompt(self) -> None:
        step = PipelineStep(
            name="test",
            model="gpt-4",
            step_type=StepType.LLM,
        )
        errors = step.validate()
        assert any("prompt_template is required" in e for e in errors)

    def test_validate_transform_step_missing_fn(self) -> None:
        step = PipelineStep(
            name="test",
            step_type=StepType.TRANSFORM,
        )
        errors = step.validate()
        assert any("transform_fn is required" in e for e in errors)

    def test_validate_parallel_group_missing_sub_steps(self) -> None:
        step = PipelineStep(
            name="test",
            step_type=StepType.PARALLEL_GROUP,
        )
        errors = step.validate()
        assert any("sub_steps required" in e for e in errors)

    def test_render_prompt(self) -> None:
        step = PipelineStep(
            name="test",
            prompt_template="Hello {name}, you are {age} years old",
        )
        result = step.render_prompt({"name": "Alice", "age": 30})
        assert result == "Hello Alice, you are 30 years old"

    def test_render_prompt_with_input_mapping(self) -> None:
        step = PipelineStep(
            name="test",
            prompt_template="Hello {user}",
            input_mapping={"user": "full_name"},
        )
        result = step.render_prompt({"full_name": "Bob"})
        assert result == "Hello Bob"

    def test_render_prompt_missing_variable(self) -> None:
        step = PipelineStep(
            name="test",
            prompt_template="Hello {missing_var}",
        )
        with pytest.raises(ValueError, match="missing variable"):
            step.render_prompt({})

    def test_to_dict(self) -> None:
        step = PipelineStep(
            name="analyze",
            model="gpt-4",
            prompt_template="Analyze: {text}",
            output_key="result",
            condition="not_empty:text",
            depends_on="previous",
        )
        d = step.to_dict()
        assert d["name"] == "analyze"
        assert d["model"] == "gpt-4"
        assert d["prompt"] == "Analyze: {text}"
        assert d["output_key"] == "result"
        assert d["condition"] == "not_empty:text"
        assert d["depends_on"] == "previous"


class TestFactoryFunctions:
    """Tests for step factory functions."""

    def test_create_llm_step(self) -> None:
        step = create_llm_step(
            name="research",
            model="gpt-4",
            prompt_template="Research {topic}",
            output_key="findings",
        )
        assert step.name == "research"
        assert step.model == "gpt-4"
        assert step.step_type == StepType.LLM
        assert step.output_key == "findings"

    def test_create_transform_step(self) -> None:
        def fn(ctx: dict) -> str:
            return ctx.get("data", "").upper()

        step = create_transform_step(name="upper", transform_fn=fn)
        assert step.name == "upper"
        assert step.step_type == StepType.TRANSFORM
        assert step.transform_fn is fn

    def test_create_parallel_group(self) -> None:
        sub1 = create_llm_step("a", "gpt-4", "prompt a")
        sub2 = create_llm_step("b", "gpt-4", "prompt b")
        group = create_parallel_group("parallel", [sub1, sub2])
        assert group.step_type == StepType.PARALLEL_GROUP
        assert len(group.sub_steps) == 2

    def test_create_conditional_step(self) -> None:
        sub = create_llm_step("inner", "gpt-4", "prompt")
        step = create_conditional_step("cond", "not_empty:data", [sub])
        assert step.step_type == StepType.CONDITIONAL
        assert step.condition == "not_empty:data"
        assert len(step.sub_steps) == 1
