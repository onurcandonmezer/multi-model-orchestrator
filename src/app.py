"""CLI demo application for the multi-model orchestrator."""

from __future__ import annotations

import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from src.executor import PipelineExecutor, mock_model_call
from src.observability import PipelineObserver
from src.pipeline import Pipeline


def _find_pipeline_files() -> list[Path]:
    """Discover YAML pipeline files in the pipelines/ directory."""
    pipelines_dir = Path(__file__).parent.parent / "pipelines"
    if not pipelines_dir.exists():
        return []
    return sorted(pipelines_dir.glob("*.yaml"))


def _display_pipeline_info(console: Console, pipeline: Pipeline) -> None:
    """Display pipeline information in a Rich panel."""
    info = Text()
    info.append("Name: ", style="bold")
    info.append(f"{pipeline.name}\n")
    info.append("Version: ", style="bold")
    info.append(f"{pipeline.version}\n")
    info.append("Description: ", style="bold")
    info.append(f"{pipeline.description}\n")
    info.append("Steps: ", style="bold")
    info.append(f"{len(pipeline.steps)}")

    console.print(Panel(info, title="Pipeline Info", border_style="cyan"))

    # Steps table
    table = Table(
        title="Pipeline Steps",
        show_header=True,
        header_style="bold magenta",
        border_style="dim",
    )
    table.add_column("#", style="dim", width=4)
    table.add_column("Step Name", style="bold")
    table.add_column("Model", style="cyan")
    table.add_column("Output Key", style="green")
    table.add_column("Depends On", style="yellow")

    for i, step in enumerate(pipeline.steps, 1):
        table.add_row(
            str(i),
            step.name,
            step.model or "N/A",
            step.output_key,
            step.depends_on or "-",
        )

    console.print(table)


def run_demo(pipeline_path: str | None = None) -> None:
    """Run the demo application.

    Args:
        pipeline_path: Optional path to a specific pipeline YAML file.
    """
    console = Console()

    console.print(Rule("[bold blue]Multi-Model Orchestrator[/]", style="blue"))
    console.print()

    # Discover or load pipeline
    if pipeline_path:
        path = Path(pipeline_path)
        if not path.exists():
            console.print(f"[red]Error: Pipeline file not found: {pipeline_path}[/]")
            return
        pipelines_to_run = [path]
    else:
        pipelines_to_run = _find_pipeline_files()
        if not pipelines_to_run:
            console.print("[yellow]No pipeline files found in pipelines/ directory[/]")
            console.print("[dim]Creating a demo pipeline...[/]")
            pipelines_to_run = []

    # If no YAML files found, run a programmatic demo
    if not pipelines_to_run:
        _run_programmatic_demo(console)
        return

    # Run each pipeline
    for pipeline_file in pipelines_to_run:
        console.print(Rule(f"[bold]{pipeline_file.stem}[/]", style="cyan"))
        console.print()

        try:
            pipeline = Pipeline.from_yaml_file(pipeline_file)
        except Exception as exc:
            console.print(f"[red]Error loading pipeline: {exc}[/]")
            continue

        _display_pipeline_info(console, pipeline)
        console.print()

        # Validate
        errors = pipeline.validate()
        if errors:
            console.print("[red]Validation errors:[/]")
            for err in errors:
                console.print(f"  [red]- {err}[/]")
            continue

        console.print("[green]Pipeline validated successfully[/]")
        console.print()

        # Execute with mock model
        observer = PipelineObserver(console=console)
        executor = PipelineExecutor(
            model_fn=mock_model_call,
            on_step_start=observer.on_step_start,
            on_step_end=observer.on_step_end,
        )

        observer.start_pipeline(pipeline)

        # Provide sample input based on pipeline
        sample_inputs = _get_sample_inputs(pipeline)
        console.print(f"[dim]Running with sample inputs: {sample_inputs}[/]")
        console.print()

        result = executor.execute(pipeline, initial_input=sample_inputs)
        observer.end_pipeline(result)

        # Display results
        observer.print_result(pipeline, result)
        console.print()

        # Show final output
        if result.final_output:
            console.print(
                Panel(
                    str(result.final_output),
                    title="Final Pipeline Output",
                    border_style="bold green",
                    width=100,
                )
            )
        console.print()


def _run_programmatic_demo(console: Console) -> None:
    """Run a demo with a programmatically built pipeline."""
    from src.steps import create_llm_step, create_transform_step

    console.print("[bold]Building demo pipeline programmatically...[/]")
    console.print()

    pipeline = Pipeline(
        name="demo_pipeline",
        description="A demo pipeline built in Python code",
        version="1.0",
    )

    pipeline.add_step(
        create_llm_step(
            name="analyze",
            model="gemini-2.5-flash-lite",
            prompt_template="Analyze the following text:\n{input_text}",
            output_key="analysis",
        )
    )

    pipeline.add_step(
        create_transform_step(
            name="format",
            transform_fn=lambda ctx: f"Formatted: {str(ctx.get('analysis', ''))[:50]}",
            output_key="formatted",
            depends_on="analyze",
        )
    )

    pipeline.add_step(
        create_llm_step(
            name="summarize",
            model="gemini-2.5-flash-lite",
            prompt_template="Summarize this analysis:\n{formatted}",
            output_key="summary",
            depends_on="format",
        )
    )

    _display_pipeline_info(console, pipeline)
    console.print()

    observer = PipelineObserver(console=console)
    executor = PipelineExecutor(
        model_fn=mock_model_call,
        on_step_start=observer.on_step_start,
        on_step_end=observer.on_step_end,
    )

    observer.start_pipeline(pipeline)
    result = executor.execute(
        pipeline,
        initial_input={"input_text": "The multi-model orchestrator enables complex AI workflows."},
    )
    observer.end_pipeline(result)

    observer.print_result(pipeline, result)


def _get_sample_inputs(pipeline: Pipeline) -> dict[str, str]:
    """Generate sample inputs based on pipeline step prompts."""
    samples: dict[str, str] = {}

    if not pipeline.steps:
        return samples

    first_step = pipeline.steps[0]
    prompt = first_step.prompt_template

    # Extract variable names from the first step's prompt
    import re

    variables = re.findall(r"\{(\w+)\}", prompt)

    for var in variables:
        # Check if this variable is an output of another step
        is_step_output = any(s.output_key == var for s in pipeline.steps)
        if not is_step_output:
            samples[var] = f"Sample input for {var}"

    if not samples:
        samples["input"] = "Sample input data for testing"

    return samples


def main() -> None:
    """Entry point for the CLI application."""
    pipeline_path = sys.argv[1] if len(sys.argv) > 1 else None
    run_demo(pipeline_path)


if __name__ == "__main__":
    main()
