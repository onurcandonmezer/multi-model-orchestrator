"""Error handling and retry strategies for the multi-model orchestrator."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class BackoffStrategy(StrEnum):
    """Backoff strategies for retries."""

    CONSTANT = "constant"
    EXPONENTIAL = "exponential"


@dataclass
class RetryStrategy:
    """Configuration for retry behavior.

    Attributes:
        max_retries: Maximum number of retry attempts.
        backoff: Backoff strategy between retries.
        base_delay_seconds: Base delay between retries.
        max_delay_seconds: Maximum delay between retries.
        retry_on: Exception types to retry on. If empty, retries on all exceptions.
    """

    max_retries: int = 3
    backoff: BackoffStrategy = BackoffStrategy.EXPONENTIAL
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    retry_on: tuple[type[Exception], ...] = ()

    def get_delay(self, attempt: int) -> float:
        """Calculate the delay for a given attempt number (0-indexed)."""
        if self.backoff == BackoffStrategy.CONSTANT:
            delay = self.base_delay_seconds
        else:
            delay = self.base_delay_seconds * (2**attempt)
        return min(delay, self.max_delay_seconds)

    def should_retry(self, exception: Exception) -> bool:
        """Determine if the given exception should trigger a retry."""
        if not self.retry_on:
            return True
        return isinstance(exception, self.retry_on)


def with_retry[T](
    fn: Callable[..., T],
    strategy: RetryStrategy,
    *args: Any,
    sleep_fn: Callable[[float], None] = time.sleep,
    **kwargs: Any,
) -> T:
    """Execute a function with retry logic.

    Args:
        fn: The function to execute.
        strategy: The retry strategy to use.
        *args: Positional arguments to pass to the function.
        sleep_fn: Sleep function (injectable for testing).
        **kwargs: Keyword arguments to pass to the function.

    Returns:
        The return value of the function.

    Raises:
        The last exception if all retries are exhausted.
    """
    last_exception: Exception | None = None

    for attempt in range(strategy.max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last_exception = exc
            if not strategy.should_retry(exc):
                raise
            if attempt < strategy.max_retries:
                delay = strategy.get_delay(attempt)
                sleep_fn(delay)

    raise last_exception  # type: ignore[misc]


@dataclass
class FallbackChain:
    """Try multiple models/functions in sequence until one succeeds.

    Attributes:
        options: List of (name, callable) pairs to try in order.
    """

    options: list[tuple[str, Callable[..., Any]]] = field(default_factory=list)

    def add_option(self, name: str, fn: Callable[..., Any]) -> FallbackChain:
        """Add a fallback option. Returns self for chaining."""
        self.options.append((name, fn))
        return self

    def execute(self, *args: Any, **kwargs: Any) -> tuple[str, Any]:
        """Execute the fallback chain.

        Tries each option in order. Returns a tuple of (name, result)
        for the first option that succeeds.

        Raises:
            RuntimeError: If all options fail.
        """
        if not self.options:
            raise RuntimeError("FallbackChain has no options configured")

        errors: list[tuple[str, Exception]] = []

        for name, fn in self.options:
            try:
                result = fn(*args, **kwargs)
                return name, result
            except Exception as exc:
                errors.append((name, exc))

        error_details = "; ".join(f"{name}: {exc}" for name, exc in errors)
        raise RuntimeError(f"All fallback options failed: {error_details}")


class CircuitBreaker:
    """Stop retrying after N consecutive failures.

    Implements the circuit breaker pattern to prevent
    repeated calls to a failing service.

    States:
        - CLOSED: Normal operation, calls pass through.
        - OPEN: Too many failures, calls are rejected immediately.
        - HALF_OPEN: After recovery_timeout, allow one test call.
    """

    class State(StrEnum):
        CLOSED = "closed"
        OPEN = "open"
        HALF_OPEN = "half_open"

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout_seconds: float = 60.0,
        time_fn: Callable[[], float] = time.time,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout_seconds = recovery_timeout_seconds
        self._time_fn = time_fn
        self._state = self.State.CLOSED
        self._failure_count = 0
        self._last_failure_time: float | None = None
        self._success_count = 0

    @property
    def state(self) -> State:
        """Get the current circuit breaker state."""
        if self._state == self.State.OPEN:
            if self._last_failure_time is not None:
                elapsed = self._time_fn() - self._last_failure_time
                if elapsed >= self.recovery_timeout_seconds:
                    self._state = self.State.HALF_OPEN
        return self._state

    @property
    def failure_count(self) -> int:
        return self._failure_count

    def call[T](self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Execute a function through the circuit breaker.

        Args:
            fn: The function to execute.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            The function's return value.

        Raises:
            RuntimeError: If the circuit is open.
        """
        current_state = self.state

        if current_state == self.State.OPEN:
            raise RuntimeError(
                f"Circuit breaker is OPEN after {self._failure_count} consecutive failures"
            )

        try:
            result = fn(*args, **kwargs)
            self._on_success()
            return result
        except Exception:
            self._on_failure()
            raise

    def _on_success(self) -> None:
        """Record a successful call."""
        self._failure_count = 0
        self._success_count += 1
        self._state = self.State.CLOSED

    def _on_failure(self) -> None:
        """Record a failed call."""
        self._failure_count += 1
        self._last_failure_time = self._time_fn()
        if self._failure_count >= self.failure_threshold:
            self._state = self.State.OPEN

    def reset(self) -> None:
        """Reset the circuit breaker to its initial state."""
        self._state = self.State.CLOSED
        self._failure_count = 0
        self._last_failure_time = None
        self._success_count = 0
