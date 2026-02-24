"""Tests for the pipeline executor."""

from __future__ import annotations

from typing import Any

from src.executor import (
    AsyncPipelineExecutor,
    PipelineExecutor,
    _evaluate_condition,
    mock_model_call,
)
from src.pipeline import Pipeline
from src.steps import (
    PipelineStep,
    create_llm_step,
    create_transform_step,
)


class TestMockModelCall:
    """Tests for the mock model callable."""

    def test_short_prompt(self) -> None:
        result = mock_model_call("gpt-4", "Hello world", {})
        assert "[Mock gpt-4 response]" in result
        assert "Hello world" in result

    def test_long_prompt(self) -> None:
        long_prompt = "x" * 200
        result = mock_model_call("gpt-4", long_prompt, {})
        assert "[Mock gpt-4 response]" in result
        assert "..." in result


class TestEvaluateCondition:
    """Tests for condition evaluation."""

    def test_empty_condition(self) -> None:
        assert _evaluate_condition("", {}) is True

    def test_key_exists_true(self) -> None:
        assert _evaluate_condition("key_exists:data", {"data": "value"}) is True

    def test_key_exists_false(self) -> None:
        assert _evaluate_condition("key_exists:data", {}) is False

    def test_not_empty_true(self) -> None:
        assert _evaluate_condition("not_empty:data", {"data": "value"}) is True

    def test_not_empty_false(self) -> None:
        assert _evaluate_condition("not_empty:data", {"data": ""}) is False

    def test_equals_true(self) -> None:
        assert _evaluate_condition("equals:status:ready", {"status": "ready"}) is True

    def test_equals_false(self) -> None:
        assert _evaluate_condition("equals:status:ready", {"status": "pending"}) is False

    def test_contains_true(self) -> None:
        assert _evaluate_condition("contains:text:hello", {"text": "say hello world"}) is True

    def test_contains_false(self) -> None:
        assert _evaluate_condition("contains:text:bye", {"text": "hello"}) is False


class TestPipelineExecutor:
    """Tests for the synchronous pipeline executor."""

    def _simple_pipeline(self) -> Pipeline:
        return Pipeline(
            name="test",
            steps=[
                create_llm_step("step1", "gpt-4", "Analyze: {topic}", "analysis"),
                create_llm_step(
                    "step2",
                    "gpt-4",
                    "Summarize: {analysis}",
                    "summary",
                    depends_on="step1",
                ),
            ],
        )

    def test_execute_simple_pipeline(self) -> None:
        pipeline = self._simple_pipeline()
        executor = PipelineExecutor()
        result = executor.execute(pipeline, {"topic": "AI"})

        assert result.success is True
        assert len(result.steps_results) == 2
        assert result.total_tokens > 0
        assert result.total_latency_ms > 0
        assert result.final_output is not None

    def test_execute_with_custom_model_fn(self) -> None:
        def custom_fn(model: str, prompt: str, ctx: dict[str, Any]) -> str:
            return f"CUSTOM: {model}"

        pipeline = self._simple_pipeline()
        executor = PipelineExecutor(model_fn=custom_fn)
        result = executor.execute(pipeline, {"topic": "test"})

        assert result.success is True
        assert "CUSTOM: gpt-4" in result.steps_results[0].output

    def test_execute_with_failing_model(self) -> None:
        call_count = 0

        def failing_fn(model: str, prompt: str, ctx: dict[str, Any]) -> str:
            nonlocal call_count
            call_count += 1
            raise RuntimeError("Model unavailable")

        pipeline = Pipeline(
            name="test",
            steps=[create_llm_step("step1", "gpt-4", "Hello")],
        )
        executor = PipelineExecutor(model_fn=failing_fn)
        result = executor.execute(pipeline, {})

        assert result.success is False
        assert "failed" in (result.error or "").lower()

    def test_execute_with_condition_skip(self) -> None:
        pipeline = Pipeline(
            name="test",
            steps=[
                PipelineStep(
                    name="conditional",
                    model="gpt-4",
                    prompt_template="Hello",
                    condition="key_exists:nonexistent",
                ),
            ],
        )
        executor = PipelineExecutor()
        result = executor.execute(pipeline, {})

        assert result.success is True
        assert result.steps_results[0].skipped is True

    def test_execute_with_transform_step(self) -> None:
        pipeline = Pipeline(
            name="test",
            steps=[
                create_llm_step("step1", "gpt-4", "Hello {input}", "raw"),
                create_transform_step(
                    "transform",
                    lambda ctx: str(ctx.get("raw", "")).upper(),
                    "transformed",
                ),
            ],
        )
        executor = PipelineExecutor()
        result = executor.execute(pipeline, {"input": "world"})

        assert result.success is True
        assert len(result.steps_results) == 2

    def test_execute_step_callbacks(self) -> None:
        started: list[str] = []
        ended: list[str] = []

        def on_start(step: PipelineStep) -> None:
            started.append(step.name)

        def on_end(step: PipelineStep, result: Any) -> None:
            ended.append(step.name)

        pipeline = self._simple_pipeline()
        executor = PipelineExecutor(on_step_start=on_start, on_step_end=on_end)
        executor.execute(pipeline, {"topic": "test"})

        assert started == ["step1", "step2"]
        assert ended == ["step1", "step2"]

    def test_context_passing_between_steps(self) -> None:
        """Verify that outputs from earlier steps are available to later steps."""
        outputs: list[str] = []

        def capture_fn(model: str, prompt: str, ctx: dict[str, Any]) -> str:
            outputs.append(prompt)
            return f"output_of_{model}"

        pipeline = Pipeline(
            name="test",
            steps=[
                create_llm_step("a", "model-a", "Start: {input}", "a_result"),
                create_llm_step("b", "model-b", "Continue: {a_result}", "b_result"),
            ],
        )
        executor = PipelineExecutor(model_fn=capture_fn)
        result = executor.execute(pipeline, {"input": "go"})

        assert result.success is True
        # Second step should have received the output of the first
        assert "output_of_model-a" in outputs[1]

    def test_cost_estimation(self) -> None:
        pipeline = Pipeline(
            name="test",
            steps=[create_llm_step("s", "gpt-4", "Hello")],
        )
        executor = PipelineExecutor()
        result = executor.execute(pipeline, {})

        assert result.total_cost > 0


class TestAsyncPipelineExecutor:
    """Tests for the async pipeline executor."""

    def test_async_execute(self) -> None:
        import anyio

        pipeline = Pipeline(
            name="async_test",
            steps=[
                create_llm_step("step1", "gpt-4", "Hello {input}", "out1"),
                create_llm_step("step2", "gpt-4", "World {out1}", "out2"),
            ],
        )
        executor = AsyncPipelineExecutor()

        async def _run() -> None:
            result = await executor.execute(pipeline, {"input": "test"})
            assert result.success is True
            assert len(result.steps_results) == 2
            assert result.final_output is not None

        anyio.run(_run)
