"""Pipeline step definitions for the multi-model orchestrator."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class StepType(StrEnum):
    """Types of pipeline steps."""

    LLM = "llm"
    TRANSFORM = "transform"
    CONDITIONAL = "conditional"
    PARALLEL_GROUP = "parallel_group"


@dataclass
class StepResult:
    """Result of executing a single pipeline step."""

    output: Any = None
    latency_ms: float = 0.0
    tokens_used: int = 0
    model_used: str = ""
    success: bool = True
    error: str | None = None
    step_name: str = ""
    skipped: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize the step result to a dictionary."""
        return {
            "output": self.output,
            "latency_ms": round(self.latency_ms, 2),
            "tokens_used": self.tokens_used,
            "model_used": self.model_used,
            "success": self.success,
            "error": self.error,
            "step_name": self.step_name,
            "skipped": self.skipped,
        }


@dataclass
class PipelineStep:
    """A single step in the pipeline.

    Attributes:
        name: Unique name for this step.
        model: Model identifier to use (e.g. 'gpt-4', 'gemini-2.5-flash-lite').
        prompt_template: Prompt string with {variable} placeholders.
        input_mapping: Maps context keys to template variables.
        output_key: Key to store the result under in the pipeline context.
        step_type: Type of the step (LLM, Transform, etc.).
        condition: Optional condition expression to evaluate before running.
        timeout_seconds: Maximum execution time for this step.
        max_retries: Maximum number of retry attempts on failure.
        depends_on: Name of a step that must complete before this one.
        transform_fn: Python callable for TransformStep type.
        sub_steps: Sub-steps for ParallelGroup or ConditionalStep.
    """

    name: str
    model: str = ""
    prompt_template: str = ""
    input_mapping: dict[str, str] = field(default_factory=dict)
    output_key: str = ""
    step_type: StepType = StepType.LLM
    condition: str | None = None
    timeout_seconds: float = 30.0
    max_retries: int = 0
    depends_on: str | None = None
    transform_fn: Callable[..., Any] | None = None
    sub_steps: list[PipelineStep] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.output_key:
            self.output_key = f"{self.name}_output"

    def validate(self) -> list[str]:
        """Validate the step configuration. Returns a list of error messages."""
        errors: list[str] = []
        if not self.name:
            errors.append("Step name is required")
        if self.step_type == StepType.LLM:
            if not self.model:
                errors.append(f"Step '{self.name}': model is required for LLM steps")
            if not self.prompt_template:
                errors.append(f"Step '{self.name}': prompt_template is required for LLM steps")
        elif self.step_type == StepType.TRANSFORM:
            if self.transform_fn is None:
                errors.append(f"Step '{self.name}': transform_fn is required for Transform steps")
        elif self.step_type == StepType.PARALLEL_GROUP:
            if not self.sub_steps:
                errors.append(
                    f"Step '{self.name}': sub_steps required for ParallelGroup steps"
                )
        return errors

    def render_prompt(self, context: dict[str, Any]) -> str:
        """Render the prompt template with values from the context."""
        prompt = self.prompt_template
        merged = dict(context)
        for template_var, context_key in self.input_mapping.items():
            if context_key in context:
                merged[template_var] = context[context_key]
        try:
            return prompt.format(**merged)
        except KeyError as exc:
            raise ValueError(
                f"Step '{self.name}': missing variable {exc} in prompt template"
            ) from exc

    def to_dict(self) -> dict[str, Any]:
        """Serialize the step to a dictionary."""
        result: dict[str, Any] = {
            "name": self.name,
            "step_type": self.step_type.value,
            "output_key": self.output_key,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
        }
        if self.model:
            result["model"] = self.model
        if self.prompt_template:
            result["prompt"] = self.prompt_template
        if self.input_mapping:
            result["input_mapping"] = self.input_mapping
        if self.condition:
            result["condition"] = self.condition
        if self.depends_on:
            result["depends_on"] = self.depends_on
        if self.sub_steps:
            result["sub_steps"] = [s.to_dict() for s in self.sub_steps]
        return result


def create_llm_step(
    name: str,
    model: str,
    prompt_template: str,
    output_key: str = "",
    input_mapping: dict[str, str] | None = None,
    condition: str | None = None,
    timeout_seconds: float = 30.0,
    max_retries: int = 0,
    depends_on: str | None = None,
) -> PipelineStep:
    """Factory function to create an LLM step."""
    return PipelineStep(
        name=name,
        model=model,
        prompt_template=prompt_template,
        output_key=output_key or f"{name}_output",
        input_mapping=input_mapping or {},
        step_type=StepType.LLM,
        condition=condition,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        depends_on=depends_on,
    )


def create_transform_step(
    name: str,
    transform_fn: Callable[..., Any],
    output_key: str = "",
    condition: str | None = None,
    depends_on: str | None = None,
) -> PipelineStep:
    """Factory function to create a Transform step."""
    return PipelineStep(
        name=name,
        step_type=StepType.TRANSFORM,
        output_key=output_key or f"{name}_output",
        transform_fn=transform_fn,
        condition=condition,
        depends_on=depends_on,
    )


def create_parallel_group(
    name: str,
    sub_steps: list[PipelineStep],
    output_key: str = "",
) -> PipelineStep:
    """Factory function to create a ParallelGroup step."""
    return PipelineStep(
        name=name,
        step_type=StepType.PARALLEL_GROUP,
        output_key=output_key or f"{name}_output",
        sub_steps=sub_steps,
    )


def create_conditional_step(
    name: str,
    condition: str,
    sub_steps: list[PipelineStep],
    output_key: str = "",
) -> PipelineStep:
    """Factory function to create a ConditionalStep."""
    return PipelineStep(
        name=name,
        step_type=StepType.CONDITIONAL,
        condition=condition,
        output_key=output_key or f"{name}_output",
        sub_steps=sub_steps,
    )
