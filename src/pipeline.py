"""Core pipeline engine for the multi-model orchestrator."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.steps import PipelineStep, StepResult, StepType


@dataclass
class PipelineResult:
    """Result of executing an entire pipeline."""

    steps_results: list[StepResult] = field(default_factory=list)
    total_latency_ms: float = 0.0
    total_tokens: int = 0
    total_cost: float = 0.0
    success: bool = True
    final_output: Any = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the pipeline result to a dictionary."""
        return {
            "steps_results": [r.to_dict() for r in self.steps_results],
            "total_latency_ms": round(self.total_latency_ms, 2),
            "total_tokens": self.total_tokens,
            "total_cost": round(self.total_cost, 6),
            "success": self.success,
            "final_output": self.final_output,
            "error": self.error,
        }

    @property
    def failed_steps(self) -> list[StepResult]:
        """Return a list of step results that failed."""
        return [r for r in self.steps_results if not r.success and not r.skipped]

    @property
    def successful_steps(self) -> list[StepResult]:
        """Return a list of step results that succeeded."""
        return [r for r in self.steps_results if r.success]

    @property
    def skipped_steps(self) -> list[StepResult]:
        """Return a list of step results that were skipped."""
        return [r for r in self.steps_results if r.skipped]


@dataclass
class Pipeline:
    """Declarative pipeline definition.

    A pipeline is an ordered collection of steps that are executed
    sequentially or in parallel to accomplish a complex task using
    one or more language models.
    """

    name: str
    steps: list[PipelineStep] = field(default_factory=list)
    description: str = ""
    version: str = "1.0"
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_step(self, step: PipelineStep) -> Pipeline:
        """Add a step to the pipeline. Returns self for chaining."""
        self.steps.append(step)
        return self

    def validate(self) -> list[str]:
        """Validate the pipeline configuration.

        Returns:
            A list of error messages. Empty list means valid.
        """
        errors: list[str] = []

        if not self.name:
            errors.append("Pipeline name is required")

        if not self.steps:
            errors.append("Pipeline must have at least one step")

        step_names: set[str] = set()
        output_keys: set[str] = set()

        for step in self.steps:
            # Check for duplicate step names
            if step.name in step_names:
                errors.append(f"Duplicate step name: '{step.name}'")
            step_names.add(step.name)

            # Check for duplicate output keys
            if step.output_key in output_keys:
                errors.append(f"Duplicate output key: '{step.output_key}'")
            output_keys.add(step.output_key)

            # Validate individual steps
            errors.extend(step.validate())

            # Validate depends_on references
            if step.depends_on and step.depends_on not in step_names:
                errors.append(
                    f"Step '{step.name}' depends on unknown step '{step.depends_on}'"
                )

        return errors

    def to_dict(self) -> dict[str, Any]:
        """Serialize the pipeline to a dictionary."""
        result: dict[str, Any] = {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "steps": [step.to_dict() for step in self.steps],
        }
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    def to_yaml(self) -> str:
        """Serialize the pipeline to a YAML string."""
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Pipeline:
        """Create a pipeline from a dictionary."""
        steps = []
        for step_data in data.get("steps", []):
            step = PipelineStep(
                name=step_data["name"],
                model=step_data.get("model", ""),
                prompt_template=step_data.get("prompt", step_data.get("prompt_template", "")),
                output_key=step_data.get("output_key", f"{step_data['name']}_output"),
                input_mapping=step_data.get("input_mapping", {}),
                step_type=StepType(step_data.get("step_type", "llm")),
                condition=step_data.get("condition"),
                timeout_seconds=step_data.get("timeout_seconds", 30.0),
                max_retries=step_data.get("max_retries", 0),
                depends_on=step_data.get("depends_on"),
            )
            steps.append(step)

        return cls(
            name=data["name"],
            steps=steps,
            description=data.get("description", ""),
            version=data.get("version", "1.0"),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_yaml(cls, yaml_content: str) -> Pipeline:
        """Create a pipeline from a YAML string."""
        data = yaml.safe_load(yaml_content)
        if not isinstance(data, dict):
            raise ValueError("Invalid YAML: expected a mapping at the top level")
        return cls.from_dict(data)

    @classmethod
    def from_yaml_file(cls, path: str | Path) -> Pipeline:
        """Create a pipeline from a YAML file."""
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Pipeline file not found: {path}")
        content = file_path.read_text(encoding="utf-8")
        return cls.from_yaml(content)

    def get_step(self, name: str) -> PipelineStep | None:
        """Get a step by name."""
        for step in self.steps:
            if step.name == name:
                return step
        return None

    def __len__(self) -> int:
        return len(self.steps)

    def __repr__(self) -> str:
        return (
            f"Pipeline(name='{self.name}', version='{self.version}', "
            f"steps={len(self.steps)})"
        )
