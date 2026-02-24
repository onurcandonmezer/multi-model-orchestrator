"""Tests for observability, tracing, and reporting."""

from __future__ import annotations

import json
from io import StringIO

from rich.console import Console

from src.executor import PipelineExecutor
from src.observability import PipelineObserver, PipelineTrace, StepTrace
from src.pipeline import Pipeline
from src.steps import create_llm_step


class TestStepTrace:
    """Tests for StepTrace."""

    def test_default_timestamp(self) -> None:
        trace = StepTrace(step_name="test")
        assert trace.timestamp != ""

    def test_to_dict(self) -> None:
        trace = StepTrace(
            step_name="analyze",
            model="gpt-4",
            latency_ms=123.456,
            tokens_used=50,
            cost=0.0015,
            success=True,
        )
        d = trace.to_dict()
        assert d["step_name"] == "analyze"
        assert d["model"] == "gpt-4"
        assert d["latency_ms"] == 123.46
        assert d["cost"] == 0.0015


class TestPipelineTrace:
    """Tests for PipelineTrace."""

    def test_add_step_trace(self) -> None:
        trace = PipelineTrace(pipeline_name="test")
        trace.add_step_trace(StepTrace(step_name="s1", tokens_used=10, cost=0.001))
        trace.add_step_trace(StepTrace(step_name="s2", tokens_used=20, cost=0.002))

        assert len(trace.step_traces) == 2
        assert trace.total_tokens == 30
        assert abs(trace.total_cost - 0.003) < 1e-9

    def test_finalize(self) -> None:
        trace = PipelineTrace(pipeline_name="test")
        trace.add_step_trace(StepTrace(step_name="s1", latency_ms=100, tokens_used=10))
        trace.add_step_trace(StepTrace(step_name="s2", latency_ms=200, tokens_used=20))
        trace.finalize()

        assert trace.total_latency_ms == 300
        assert trace.completed_at != ""

    def test_to_json(self) -> None:
        trace = PipelineTrace(pipeline_name="test")
        trace.add_step_trace(StepTrace(step_name="s1", tokens_used=10))
        trace.finalize()

        json_str = trace.to_json()
        data = json.loads(json_str)
        assert data["pipeline_name"] == "test"
        assert len(data["step_traces"]) == 1

    def test_to_markdown(self) -> None:
        trace = PipelineTrace(pipeline_name="test", pipeline_version="2.0")
        trace.add_step_trace(
            StepTrace(step_name="s1", model="gpt-4", latency_ms=100, tokens_used=10)
        )
        trace.finalize()

        md = trace.to_markdown()
        assert "# Pipeline Trace: test" in md
        assert "**Version:** 2.0" in md
        assert "| s1 |" in md

    def test_to_markdown_with_errors(self) -> None:
        trace = PipelineTrace(pipeline_name="test")
        trace.add_step_trace(StepTrace(step_name="s1", success=False, error="Something broke"))
        trace.finalize()

        md = trace.to_markdown()
        assert "Something broke" in md

    def test_failed_trace_status(self) -> None:
        trace = PipelineTrace(pipeline_name="test")
        trace.add_step_trace(StepTrace(step_name="s1", success=False))
        assert trace.success is False


class TestPipelineObserver:
    """Tests for PipelineObserver."""

    def _make_pipeline_and_executor(self) -> tuple[Pipeline, PipelineExecutor, PipelineObserver]:
        pipeline = Pipeline(
            name="observable_test",
            version="1.0",
            steps=[
                create_llm_step("step1", "gpt-4", "Hello {input}", "out1"),
                create_llm_step("step2", "gpt-4", "World {out1}", "out2"),
            ],
        )
        observer = PipelineObserver(console=Console(file=StringIO()))
        executor = PipelineExecutor(
            on_step_start=observer.on_step_start,
            on_step_end=observer.on_step_end,
        )
        return pipeline, executor, observer

    def test_observer_collects_traces(self) -> None:
        pipeline, executor, observer = self._make_pipeline_and_executor()

        observer.start_pipeline(pipeline)
        result = executor.execute(pipeline, {"input": "test"})
        observer.end_pipeline(result)

        assert len(observer.traces) == 1
        trace = observer.traces[0]
        assert trace.pipeline_name == "observable_test"
        assert len(trace.step_traces) == 2

    def test_observer_print_result(self) -> None:
        """Verify print_result does not raise."""
        pipeline, executor, observer = self._make_pipeline_and_executor()

        observer.start_pipeline(pipeline)
        result = executor.execute(pipeline, {"input": "test"})
        observer.end_pipeline(result)

        # Should not raise
        observer.print_result(pipeline, result)

    def test_build_trace_from_result(self) -> None:
        pipeline = Pipeline(
            name="trace_test",
            steps=[create_llm_step("s1", "gpt-4", "Hello")],
        )
        observer = PipelineObserver(console=Console(file=StringIO()))
        executor = PipelineExecutor()
        result = executor.execute(pipeline, {})

        trace = observer.build_trace_from_result(pipeline, result)
        assert trace.pipeline_name == "trace_test"
        assert len(trace.step_traces) == 1
        assert trace.step_traces[0].step_name == "s1"
