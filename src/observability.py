"""Logging, metrics, and tracing for the multi-model orchestrator."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.pipeline import Pipeline, PipelineResult
from src.steps import PipelineStep, StepResult


@dataclass
class StepTrace:
    """Trace data for a single step execution."""

    step_name: str
    model: str = ""
    input_preview: str = ""
    output_preview: str = ""
    latency_ms: float = 0.0
    tokens_used: int = 0
    cost: float = 0.0
    success: bool = True
    error: str | None = None
    skipped: bool = False
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(UTC).isoformat()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "step_name": self.step_name,
            "model": self.model,
            "input_preview": self.input_preview,
            "output_preview": self.output_preview,
            "latency_ms": round(self.latency_ms, 2),
            "tokens_used": self.tokens_used,
            "cost": round(self.cost, 6),
            "success": self.success,
            "error": self.error,
            "skipped": self.skipped,
            "timestamp": self.timestamp,
        }


@dataclass
class PipelineTrace:
    """Trace data for an entire pipeline execution."""

    pipeline_name: str
    pipeline_version: str = "1.0"
    step_traces: list[StepTrace] = field(default_factory=list)
    total_latency_ms: float = 0.0
    total_tokens: int = 0
    total_cost: float = 0.0
    success: bool = True
    started_at: str = ""
    completed_at: str = ""

    def __post_init__(self) -> None:
        if not self.started_at:
            self.started_at = datetime.now(UTC).isoformat()

    def add_step_trace(self, trace: StepTrace) -> None:
        """Add a step trace to the pipeline trace."""
        self.step_traces.append(trace)
        self.total_tokens += trace.tokens_used
        self.total_cost += trace.cost
        if not trace.success and not trace.skipped:
            self.success = False

    def finalize(self) -> None:
        """Finalize the trace with computed totals."""
        self.completed_at = datetime.now(UTC).isoformat()
        if self.step_traces:
            self.total_latency_ms = sum(t.latency_ms for t in self.step_traces)
            self.total_tokens = sum(t.tokens_used for t in self.step_traces)
            self.total_cost = sum(t.cost for t in self.step_traces)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "pipeline_name": self.pipeline_name,
            "pipeline_version": self.pipeline_version,
            "step_traces": [t.to_dict() for t in self.step_traces],
            "total_latency_ms": round(self.total_latency_ms, 2),
            "total_tokens": self.total_tokens,
            "total_cost": round(self.total_cost, 6),
            "success": self.success,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }

    def to_json(self, indent: int = 2) -> str:
        """Export trace as JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def to_markdown(self) -> str:
        """Export trace as a Markdown report."""
        lines = [
            f"# Pipeline Trace: {self.pipeline_name}",
            "",
            f"**Version:** {self.pipeline_version}",
            f"**Status:** {'Success' if self.success else 'Failed'}",
            f"**Total Latency:** {self.total_latency_ms:.2f}ms",
            f"**Total Tokens:** {self.total_tokens}",
            f"**Total Cost:** ${self.total_cost:.6f}",
            f"**Started:** {self.started_at}",
            f"**Completed:** {self.completed_at}",
            "",
            "## Steps",
            "",
            "| Step | Model | Latency (ms) | Tokens | Cost | Status |",
            "|------|-------|-------------|--------|------|--------|",
        ]

        for trace in self.step_traces:
            status = "Skipped" if trace.skipped else ("Pass" if trace.success else "Fail")
            lines.append(
                f"| {trace.step_name} | {trace.model or 'N/A'} "
                f"| {trace.latency_ms:.2f} | {trace.tokens_used} "
                f"| ${trace.cost:.6f} | {status} |"
            )

        lines.append("")

        for trace in self.step_traces:
            if trace.error:
                lines.append(f"### Error in `{trace.step_name}`")
                lines.append(f"```\n{trace.error}\n```")
                lines.append("")

        return "\n".join(lines)


# Default cost per 1K tokens for cost estimation
_COST_PER_1K: dict[str, float] = {
    "gpt-4": 0.03,
    "gpt-4-turbo": 0.01,
    "gpt-3.5-turbo": 0.0015,
    "claude-3-opus": 0.015,
    "claude-3-sonnet": 0.003,
    "claude-3-haiku": 0.00025,
    "gemini-2.5-flash-lite": 0.0001,
    "gemini-2.0-flash": 0.0005,
}


def _estimate_cost(model: str, tokens: int) -> float:
    """Estimate cost for a model call."""
    rate = _COST_PER_1K.get(model, 0.001)
    return (tokens / 1000) * rate


class PipelineObserver:
    """Observes pipeline execution and collects traces.

    Can be used as step callbacks in the PipelineExecutor to
    automatically trace all step executions.
    """

    def __init__(self, console: Console | None = None) -> None:
        self._console = console or Console()
        self._traces: list[PipelineTrace] = []
        self._current_trace: PipelineTrace | None = None
        self._step_inputs: dict[str, str] = {}

    @property
    def traces(self) -> list[PipelineTrace]:
        """Get all collected pipeline traces."""
        return list(self._traces)

    @property
    def current_trace(self) -> PipelineTrace | None:
        """Get the current (in-progress) trace."""
        return self._current_trace

    def start_pipeline(self, pipeline: Pipeline) -> None:
        """Begin tracing a pipeline execution."""
        self._current_trace = PipelineTrace(
            pipeline_name=pipeline.name,
            pipeline_version=pipeline.version,
        )

    def on_step_start(self, step: PipelineStep) -> None:
        """Called when a step begins execution."""
        self._step_inputs[step.name] = step.prompt_template[:100] if step.prompt_template else ""

    def on_step_end(self, step: PipelineStep, result: StepResult) -> None:
        """Called when a step completes execution."""
        if self._current_trace is None:
            return

        output_preview = str(result.output)[:100] if result.output else ""
        input_preview = self._step_inputs.get(step.name, "")

        cost = _estimate_cost(result.model_used, result.tokens_used) if result.model_used else 0.0

        trace = StepTrace(
            step_name=step.name,
            model=result.model_used,
            input_preview=input_preview,
            output_preview=output_preview,
            latency_ms=result.latency_ms,
            tokens_used=result.tokens_used,
            cost=cost,
            success=result.success,
            error=result.error,
            skipped=result.skipped,
        )

        self._current_trace.add_step_trace(trace)

    def end_pipeline(self, result: PipelineResult) -> None:
        """Finalize the current pipeline trace."""
        if self._current_trace is None:
            return

        self._current_trace.total_latency_ms = result.total_latency_ms
        self._current_trace.success = result.success
        self._current_trace.finalize()
        self._traces.append(self._current_trace)
        self._current_trace = None

    def build_trace_from_result(
        self, pipeline: Pipeline, result: PipelineResult
    ) -> PipelineTrace:
        """Build a PipelineTrace from a completed PipelineResult."""
        trace = PipelineTrace(
            pipeline_name=pipeline.name,
            pipeline_version=pipeline.version,
            total_latency_ms=result.total_latency_ms,
            total_tokens=result.total_tokens,
            total_cost=result.total_cost,
            success=result.success,
        )

        for sr in result.steps_results:
            cost = _estimate_cost(sr.model_used, sr.tokens_used) if sr.model_used else 0.0
            step_trace = StepTrace(
                step_name=sr.step_name,
                model=sr.model_used,
                output_preview=str(sr.output)[:100] if sr.output else "",
                latency_ms=sr.latency_ms,
                tokens_used=sr.tokens_used,
                cost=cost,
                success=sr.success,
                error=sr.error,
                skipped=sr.skipped,
            )
            trace.step_traces.append(step_trace)

        trace.finalize()
        return trace

    def print_result(self, pipeline: Pipeline, result: PipelineResult) -> None:
        """Pretty print pipeline execution results using Rich."""
        trace = self.build_trace_from_result(pipeline, result)
        self._print_trace(trace)

    def _print_trace(self, trace: PipelineTrace) -> None:
        """Pretty print a pipeline trace using Rich tables and panels."""
        console = self._console

        # Header panel
        status_text = "[bold green]SUCCESS[/]" if trace.success else "[bold red]FAILED[/]"
        header = Text.from_markup(
            f"[bold]{trace.pipeline_name}[/] v{trace.pipeline_version}\n"
            f"Status: {status_text}\n"
            f"Total Latency: {trace.total_latency_ms:.2f}ms | "
            f"Tokens: {trace.total_tokens} | "
            f"Cost: ${trace.total_cost:.6f}"
        )
        console.print(Panel(header, title="Pipeline Execution", border_style="blue"))

        # Steps table
        table = Table(
            title="Step Results",
            show_header=True,
            header_style="bold cyan",
            border_style="dim",
        )
        table.add_column("Step", style="bold")
        table.add_column("Model", style="dim")
        table.add_column("Latency (ms)", justify="right")
        table.add_column("Tokens", justify="right")
        table.add_column("Cost", justify="right")
        table.add_column("Status", justify="center")

        for step_trace in trace.step_traces:
            if step_trace.skipped:
                status = "[dim]SKIPPED[/]"
            elif step_trace.success:
                status = "[green]PASS[/]"
            else:
                status = "[red]FAIL[/]"

            table.add_row(
                step_trace.step_name,
                step_trace.model or "N/A",
                f"{step_trace.latency_ms:.2f}",
                str(step_trace.tokens_used),
                f"${step_trace.cost:.6f}",
                status,
            )

        console.print(table)

        # Show errors if any
        for step_trace in trace.step_traces:
            if step_trace.error:
                console.print(
                    Panel(
                        f"[red]{step_trace.error}[/]",
                        title=f"Error: {step_trace.step_name}",
                        border_style="red",
                    )
                )

        # Output preview for successful traces
        for step_trace in trace.step_traces:
            if step_trace.output_preview and step_trace.success and not step_trace.skipped:
                console.print(
                    Panel(
                        step_trace.output_preview,
                        title=f"Output: {step_trace.step_name}",
                        border_style="green",
                        width=100,
                    )
                )
