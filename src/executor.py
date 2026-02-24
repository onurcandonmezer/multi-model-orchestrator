"""Sequential and parallel execution engine for the multi-model orchestrator."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import anyio

from src.pipeline import Pipeline, PipelineResult
from src.retry import RetryStrategy, with_retry
from src.steps import PipelineStep, StepResult, StepType

# Default cost per 1K tokens by model (simplified estimates)
DEFAULT_COST_PER_1K_TOKENS: dict[str, float] = {
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
    """Estimate cost based on model and token count."""
    rate = DEFAULT_COST_PER_1K_TOKENS.get(model, 0.001)
    return (tokens / 1000) * rate


ModelCallable = Callable[[str, str, dict[str, Any]], str]
"""Type alias for a model callable: (model_name, prompt, context) -> response."""


def mock_model_call(model: str, prompt: str, context: dict[str, Any]) -> str:
    """A mock model call that returns a simulated response.

    This is useful for testing pipelines without actual API calls.
    """
    prompt_preview = prompt[:80].replace("\n", " ")
    return (
        f"[Mock {model} response] Processed: '{prompt_preview}...'"
        if len(prompt) > 80
        else f"[Mock {model} response] Processed: '{prompt.replace(chr(10), ' ')}'"
    )


def _evaluate_condition(condition: str, context: dict[str, Any]) -> bool:
    """Evaluate a condition string against the pipeline context.

    Supports simple conditions like:
      - 'key_exists:variable_name' - check if a key exists
      - 'not_empty:variable_name' - check if a value is truthy
      - 'equals:variable_name:value' - check equality
      - 'contains:variable_name:substring' - check substring
    """
    if not condition:
        return True

    parts = condition.split(":")
    op = parts[0].strip().lower()

    if op == "key_exists" and len(parts) >= 2:
        key = parts[1].strip()
        return key in context

    if op == "not_empty" and len(parts) >= 2:
        key = parts[1].strip()
        return bool(context.get(key))

    if op == "equals" and len(parts) >= 3:
        key = parts[1].strip()
        value = parts[2].strip()
        return str(context.get(key, "")) == value

    if op == "contains" and len(parts) >= 3:
        key = parts[1].strip()
        substring = parts[2].strip()
        return substring in str(context.get(key, ""))

    # Default: treat as truthy check on the condition string itself
    return bool(context.get(condition))


@dataclass
class PipelineExecutor:
    """Executes a pipeline by running each step in sequence or parallel.

    Attributes:
        model_fn: Callable that simulates or calls a model.
        retry_strategy: Default retry strategy for all steps.
        on_step_start: Optional callback invoked when a step starts.
        on_step_end: Optional callback invoked when a step completes.
    """

    model_fn: ModelCallable = mock_model_call
    retry_strategy: RetryStrategy | None = None
    on_step_start: Callable[[PipelineStep], None] | None = None
    on_step_end: Callable[[PipelineStep, StepResult], None] | None = None

    def execute(
        self,
        pipeline: Pipeline,
        initial_input: dict[str, Any] | None = None,
    ) -> PipelineResult:
        """Execute a pipeline synchronously.

        Args:
            pipeline: The pipeline to execute.
            initial_input: Initial context variables.

        Returns:
            PipelineResult with all step results and metrics.
        """
        context: dict[str, Any] = dict(initial_input or {})
        result = PipelineResult()
        pipeline_start = time.perf_counter()

        for step in pipeline.steps:
            step_result = self._execute_step(step, context)
            result.steps_results.append(step_result)

            if step_result.success and not step_result.skipped:
                context[step.output_key] = step_result.output
            elif not step_result.success:
                result.success = False
                result.error = f"Step '{step.name}' failed: {step_result.error}"
                break

        pipeline_end = time.perf_counter()
        result.total_latency_ms = (pipeline_end - pipeline_start) * 1000

        for sr in result.steps_results:
            result.total_tokens += sr.tokens_used
            if sr.model_used:
                result.total_cost += _estimate_cost(sr.model_used, sr.tokens_used)

        if result.steps_results and result.success:
            last_successful = [
                r for r in result.steps_results if r.success and not r.skipped
            ]
            if last_successful:
                result.final_output = last_successful[-1].output

        return result

    def _execute_step(
        self, step: PipelineStep, context: dict[str, Any]
    ) -> StepResult:
        """Execute a single step."""
        # Check condition
        if step.condition and not _evaluate_condition(step.condition, context):
            return StepResult(
                step_name=step.name,
                skipped=True,
                success=True,
                output=None,
            )

        if self.on_step_start:
            self.on_step_start(step)

        start_time = time.perf_counter()

        try:
            if step.step_type == StepType.LLM:
                step_result = self._execute_llm_step(step, context)
            elif step.step_type == StepType.TRANSFORM:
                step_result = self._execute_transform_step(step, context)
            elif step.step_type == StepType.PARALLEL_GROUP:
                step_result = self._execute_parallel_group(step, context)
            elif step.step_type == StepType.CONDITIONAL:
                step_result = self._execute_conditional_step(step, context)
            else:
                step_result = StepResult(
                    step_name=step.name,
                    success=False,
                    error=f"Unknown step type: {step.step_type}",
                )
        except Exception as exc:
            step_result = StepResult(
                step_name=step.name,
                success=False,
                error=str(exc),
            )

        end_time = time.perf_counter()
        step_result.latency_ms = (end_time - start_time) * 1000
        step_result.step_name = step.name

        if self.on_step_end:
            self.on_step_end(step, step_result)

        return step_result

    def _execute_llm_step(
        self, step: PipelineStep, context: dict[str, Any]
    ) -> StepResult:
        """Execute an LLM step with optional retry."""
        prompt = step.render_prompt(context)

        def _call() -> str:
            return self.model_fn(step.model, prompt, context)

        strategy = self.retry_strategy
        if step.max_retries > 0:
            strategy = RetryStrategy(
                max_retries=step.max_retries,
                base_delay_seconds=0.1,
            )

        if strategy and strategy.max_retries > 0:
            output = with_retry(_call, strategy, sleep_fn=lambda _: None)
        else:
            output = _call()

        tokens = max(len(prompt.split()) + len(str(output).split()), 10)

        return StepResult(
            output=output,
            tokens_used=tokens,
            model_used=step.model,
            success=True,
        )

    def _execute_transform_step(
        self, step: PipelineStep, context: dict[str, Any]
    ) -> StepResult:
        """Execute a Transform step."""
        if step.transform_fn is None:
            return StepResult(
                step_name=step.name,
                success=False,
                error="No transform function provided",
            )

        output = step.transform_fn(context)
        return StepResult(
            output=output,
            success=True,
        )

    def _execute_parallel_group(
        self, step: PipelineStep, context: dict[str, Any]
    ) -> StepResult:
        """Execute a group of steps in parallel using anyio."""
        sub_results: list[StepResult] = []

        async def _run_parallel() -> None:
            async with anyio.create_task_group() as tg:
                for sub_step in step.sub_steps:
                    async def _run(s: PipelineStep = sub_step) -> None:
                        r = self._execute_step(s, dict(context))
                        sub_results.append(r)
                        if r.success and not r.skipped:
                            context[s.output_key] = r.output

                    tg.start_soon(_run)

        anyio.from_thread.run_sync(lambda: anyio.run(_run_parallel))

        combined_output = {
            r.step_name: r.output for r in sub_results if r.success and not r.skipped
        }
        total_tokens = sum(r.tokens_used for r in sub_results)
        all_success = all(r.success for r in sub_results)

        return StepResult(
            output=combined_output,
            tokens_used=total_tokens,
            success=all_success,
            error=None if all_success else "One or more parallel steps failed",
        )

    def _execute_conditional_step(
        self, step: PipelineStep, context: dict[str, Any]
    ) -> StepResult:
        """Execute a conditional step (runs sub-steps if condition is met)."""
        if step.condition and not _evaluate_condition(step.condition, context):
            return StepResult(
                step_name=step.name,
                skipped=True,
                success=True,
            )

        results = []
        for sub_step in step.sub_steps:
            r = self._execute_step(sub_step, context)
            results.append(r)
            if r.success and not r.skipped:
                context[sub_step.output_key] = r.output
            if not r.success:
                return StepResult(
                    step_name=step.name,
                    success=False,
                    error=f"Sub-step '{sub_step.name}' failed: {r.error}",
                )

        last_output = results[-1].output if results else None
        total_tokens = sum(r.tokens_used for r in results)

        return StepResult(
            output=last_output,
            tokens_used=total_tokens,
            success=True,
        )


class AsyncPipelineExecutor:
    """Async version of the pipeline executor."""

    def __init__(
        self,
        model_fn: ModelCallable = mock_model_call,
        retry_strategy: RetryStrategy | None = None,
    ) -> None:
        self.model_fn = model_fn
        self.retry_strategy = retry_strategy

    async def execute(
        self,
        pipeline: Pipeline,
        initial_input: dict[str, Any] | None = None,
    ) -> PipelineResult:
        """Execute a pipeline asynchronously."""
        context: dict[str, Any] = dict(initial_input or {})
        result = PipelineResult()
        pipeline_start = time.perf_counter()

        for step in pipeline.steps:
            step_result = await self._execute_step(step, context)
            result.steps_results.append(step_result)

            if step_result.success and not step_result.skipped:
                context[step.output_key] = step_result.output
            elif not step_result.success:
                result.success = False
                result.error = f"Step '{step.name}' failed: {step_result.error}"
                break

        pipeline_end = time.perf_counter()
        result.total_latency_ms = (pipeline_end - pipeline_start) * 1000

        for sr in result.steps_results:
            result.total_tokens += sr.tokens_used
            if sr.model_used:
                result.total_cost += _estimate_cost(sr.model_used, sr.tokens_used)

        if result.steps_results and result.success:
            last_successful = [
                r for r in result.steps_results if r.success and not r.skipped
            ]
            if last_successful:
                result.final_output = last_successful[-1].output

        return result

    async def _execute_step(
        self, step: PipelineStep, context: dict[str, Any]
    ) -> StepResult:
        """Execute a single step asynchronously."""
        if step.condition and not _evaluate_condition(step.condition, context):
            return StepResult(
                step_name=step.name,
                skipped=True,
                success=True,
            )

        start_time = time.perf_counter()

        try:
            if step.step_type == StepType.LLM:
                prompt = step.render_prompt(context)
                output = self.model_fn(step.model, prompt, context)
                tokens = max(len(prompt.split()) + len(str(output).split()), 10)
                step_result = StepResult(
                    output=output,
                    tokens_used=tokens,
                    model_used=step.model,
                    success=True,
                )
            elif step.step_type == StepType.TRANSFORM:
                if step.transform_fn is None:
                    step_result = StepResult(
                        success=False, error="No transform function"
                    )
                else:
                    output = step.transform_fn(context)
                    step_result = StepResult(output=output, success=True)
            elif step.step_type == StepType.PARALLEL_GROUP:
                step_result = await self._execute_parallel(step, context)
            else:
                step_result = StepResult(
                    success=False, error=f"Unknown step type: {step.step_type}"
                )
        except Exception as exc:
            step_result = StepResult(success=False, error=str(exc))

        end_time = time.perf_counter()
        step_result.latency_ms = (end_time - start_time) * 1000
        step_result.step_name = step.name
        return step_result

    async def _execute_parallel(
        self, step: PipelineStep, context: dict[str, Any]
    ) -> StepResult:
        """Execute sub-steps in parallel."""
        sub_results: list[StepResult] = []

        async with anyio.create_task_group() as tg:
            for sub_step in step.sub_steps:
                async def _run(s: PipelineStep = sub_step) -> None:
                    r = await self._execute_step(s, dict(context))
                    sub_results.append(r)

                tg.start_soon(_run)

        combined_output = {
            r.step_name: r.output for r in sub_results if r.success and not r.skipped
        }
        total_tokens = sum(r.tokens_used for r in sub_results)
        all_success = all(r.success for r in sub_results)

        return StepResult(
            output=combined_output,
            tokens_used=total_tokens,
            success=all_success,
        )
