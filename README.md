# Multi-Model Orchestrator

[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue?logo=python&logoColor=white)](https://python.org)
[![asyncio](https://img.shields.io/badge/asyncio-supported-green?logo=python&logoColor=white)](https://docs.python.org/3/library/asyncio.html)
[![Pydantic](https://img.shields.io/badge/Pydantic-v2-E92063?logo=pydantic&logoColor=white)](https://docs.pydantic.dev/)
[![YAML](https://img.shields.io/badge/YAML-pipelines-CB171E?logo=yaml&logoColor=white)](https://yaml.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-2088FF?logo=github-actions&logoColor=white)](.github/workflows/ci.yml)

A declarative multi-LLM pipeline framework for chaining multiple language models together to accomplish complex tasks. Define pipelines in YAML or Python, execute them with built-in retry logic, fallback chains, circuit breakers, and full observability tracing.

## Key Features

- **Declarative Pipelines** -- Define multi-step LLM workflows in YAML or build them programmatically in Python
- **Multi-Model Support** -- Chain different models (GPT-4, Claude, Gemini, etc.) in a single pipeline
- **Sequential & Parallel Execution** -- Run steps in order or execute independent steps concurrently with asyncio
- **Conditional Logic** -- Skip or include steps based on runtime conditions
- **Transform Steps** -- Apply Python functions between LLM calls for data transformation
- **Retry Strategies** -- Constant or exponential backoff with configurable retry-on exception types
- **Fallback Chains** -- Automatically try alternative models when the primary model fails
- **Circuit Breaker** -- Stop retrying after consecutive failures to prevent cascading issues
- **Full Observability** -- Step-level tracing with latency, token usage, cost estimation, and export to JSON/Markdown
- **Rich CLI Output** -- Beautiful terminal output with Rich tables, panels, and progress indicators
- **Mock Execution** -- Test pipelines without API keys using built-in mock model calls

## Architecture

```
                    +------------------+
                    |   Pipeline YAML  |
                    |  or Python API   |
                    +--------+---------+
                             |
                    +--------v---------+
                    |     Pipeline     |
                    |   (validated)    |
                    +--------+---------+
                             |
                    +--------v---------+
                    | PipelineExecutor |
                    +--------+---------+
                             |
              +--------------+--------------+
              |              |              |
     +--------v---+  +------v------+  +----v--------+
     |  LLM Step  |  | Transform   |  |  Parallel   |
     |            |  |   Step      |  |   Group     |
     +--------+---+  +------+------+  +----+--------+
              |              |              |
              +--------------+--------------+
                             |
                    +--------v---------+
                    |   RetryStrategy  |
                    |  FallbackChain   |
                    |  CircuitBreaker  |
                    +--------+---------+
                             |
                    +--------v---------+
                    | PipelineObserver |
                    |  (traces, logs)  |
                    +--------+---------+
                             |
                    +--------v---------+
                    |  PipelineResult  |
                    | JSON / Markdown  |
                    +------------------+
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/onurcandonmezer/multi-model-orchestrator.git
cd multi-model-orchestrator

# Install with uv
uv venv && uv pip install -e ".[dev]"

# Run the demo
uv run python -m src.app
```

### Run Tests

```bash
uv run python -m pytest tests/ -v --tb=short
```

## YAML Pipeline Example

```yaml
name: research_and_summarize
version: "1.0"
description: "Research a topic and create a summary"
steps:
  - name: research
    model: gemini-2.5-flash-lite
    prompt: "Research the following topic:\n{topic}"
    output_key: research_output

  - name: summarize
    model: gemini-2.5-flash-lite
    prompt: "Summarize these findings:\n{research_output}"
    output_key: summary
    depends_on: research
```

## Python API Usage

```python
from src.pipeline import Pipeline
from src.executor import PipelineExecutor
from src.observability import PipelineObserver
from src.steps import create_llm_step, create_transform_step

# Build pipeline programmatically
pipeline = Pipeline(name="analysis", description="Analyze and summarize")

pipeline.add_step(
    create_llm_step(
        name="analyze",
        model="gemini-2.5-flash-lite",
        prompt_template="Analyze: {topic}",
        output_key="analysis",
    )
)

pipeline.add_step(
    create_transform_step(
        name="format",
        transform_fn=lambda ctx: ctx["analysis"].upper(),
        output_key="formatted",
    )
)

pipeline.add_step(
    create_llm_step(
        name="summarize",
        model="gemini-2.5-flash-lite",
        prompt_template="Summarize: {formatted}",
        output_key="summary",
        depends_on="format",
    )
)

# Execute with mock models (no API keys needed)
executor = PipelineExecutor()
result = executor.execute(pipeline, {"topic": "AI Safety"})

# Observe and trace
observer = PipelineObserver()
trace = observer.build_trace_from_result(pipeline, result)
print(trace.to_markdown())
```

### Retry and Fallback

```python
from src.retry import RetryStrategy, FallbackChain, CircuitBreaker, BackoffStrategy

# Retry with exponential backoff
strategy = RetryStrategy(
    max_retries=3,
    backoff=BackoffStrategy.EXPONENTIAL,
    base_delay_seconds=1.0,
)

# Fallback chain: try multiple models
chain = FallbackChain()
chain.add_option("primary", lambda: call_model("gpt-4", prompt))
chain.add_option("fallback", lambda: call_model("gpt-3.5-turbo", prompt))
name, result = chain.execute()

# Circuit breaker: stop after 5 consecutive failures
breaker = CircuitBreaker(failure_threshold=5, recovery_timeout_seconds=60)
result = breaker.call(lambda: call_model("gpt-4", prompt))
```

## Pipeline Templates

| Template | Description |
|----------|-------------|
| `research_and_summarize` | Research a topic, extract key points, and create an executive summary |
| `extract_and_validate` | Extract structured data from text and validate it for accuracy |
| `translate_and_review` | Translate text, back-translate for verification, and review quality |
| `analyze_and_report` | Analyze data, identify trends, generate recommendations, and compile a report |

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)
![Pydantic](https://img.shields.io/badge/Pydantic-v2-E92063?logo=pydantic&logoColor=white)
![PyYAML](https://img.shields.io/badge/PyYAML-6.0-CB171E)
![Rich](https://img.shields.io/badge/Rich-13.0-4B8BBE)
![anyio](https://img.shields.io/badge/anyio-4.0-green)
![pytest](https://img.shields.io/badge/pytest-8.0-0A9EDC?logo=pytest&logoColor=white)
![Ruff](https://img.shields.io/badge/Ruff-linter-D7FF64?logo=ruff&logoColor=black)

## Project Structure

```
multi-model-orchestrator/
├── README.md
├── pyproject.toml
├── Makefile
├── LICENSE
├── .gitignore
├── .github/workflows/ci.yml
├── src/
│   ├── __init__.py
│   ├── pipeline.py          # Core pipeline engine
│   ├── steps.py             # Pipeline step definitions
│   ├── executor.py          # Sequential & parallel execution
│   ├── observability.py     # Logging, metrics, tracing
│   ├── retry.py             # Error handling & retry strategies
│   └── app.py               # CLI/demo app with Rich output
├── tests/
│   ├── __init__.py
│   ├── test_pipeline.py
│   ├── test_steps.py
│   ├── test_executor.py
│   ├── test_observability.py
│   └── test_retry.py
├── pipelines/
│   ├── research_and_summarize.yaml
│   ├── extract_and_validate.yaml
│   ├── translate_and_review.yaml
│   └── analyze_and_report.yaml
└── assets/
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
