"""Tests for error handling and retry strategies."""

from __future__ import annotations

import pytest

from src.retry import (
    BackoffStrategy,
    CircuitBreaker,
    FallbackChain,
    RetryStrategy,
    with_retry,
)


class TestRetryStrategy:
    """Tests for RetryStrategy."""

    def test_constant_delay(self) -> None:
        strategy = RetryStrategy(
            backoff=BackoffStrategy.CONSTANT,
            base_delay_seconds=2.0,
        )
        assert strategy.get_delay(0) == 2.0
        assert strategy.get_delay(1) == 2.0
        assert strategy.get_delay(5) == 2.0

    def test_exponential_delay(self) -> None:
        strategy = RetryStrategy(
            backoff=BackoffStrategy.EXPONENTIAL,
            base_delay_seconds=1.0,
        )
        assert strategy.get_delay(0) == 1.0
        assert strategy.get_delay(1) == 2.0
        assert strategy.get_delay(2) == 4.0
        assert strategy.get_delay(3) == 8.0

    def test_max_delay_cap(self) -> None:
        strategy = RetryStrategy(
            backoff=BackoffStrategy.EXPONENTIAL,
            base_delay_seconds=1.0,
            max_delay_seconds=5.0,
        )
        assert strategy.get_delay(10) == 5.0

    def test_should_retry_all_exceptions(self) -> None:
        strategy = RetryStrategy()
        assert strategy.should_retry(ValueError("test")) is True
        assert strategy.should_retry(RuntimeError("test")) is True

    def test_should_retry_specific_exceptions(self) -> None:
        strategy = RetryStrategy(retry_on=(ValueError, TypeError))
        assert strategy.should_retry(ValueError("test")) is True
        assert strategy.should_retry(TypeError("test")) is True
        assert strategy.should_retry(RuntimeError("test")) is False


class TestWithRetry:
    """Tests for the with_retry function."""

    def test_success_on_first_try(self) -> None:
        strategy = RetryStrategy(max_retries=3)
        result = with_retry(lambda: 42, strategy, sleep_fn=lambda _: None)
        assert result == 42

    def test_success_after_retries(self) -> None:
        attempts = 0

        def flaky() -> str:
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise ValueError("not yet")
            return "success"

        strategy = RetryStrategy(max_retries=5, base_delay_seconds=0.0)
        result = with_retry(flaky, strategy, sleep_fn=lambda _: None)
        assert result == "success"
        assert attempts == 3

    def test_all_retries_exhausted(self) -> None:
        def always_fail() -> None:
            raise RuntimeError("permanent failure")

        strategy = RetryStrategy(max_retries=2, base_delay_seconds=0.0)
        with pytest.raises(RuntimeError, match="permanent failure"):
            with_retry(always_fail, strategy, sleep_fn=lambda _: None)

    def test_no_retry_on_unmatched_exception(self) -> None:
        attempts = 0

        def fail_with_type_error() -> None:
            nonlocal attempts
            attempts += 1
            raise TypeError("wrong type")

        strategy = RetryStrategy(max_retries=3, retry_on=(ValueError,))
        with pytest.raises(TypeError):
            with_retry(fail_with_type_error, strategy, sleep_fn=lambda _: None)
        assert attempts == 1  # No retries attempted

    def test_sleep_fn_called(self) -> None:
        delays: list[float] = []
        attempts = 0

        def fail_twice() -> str:
            nonlocal attempts
            attempts += 1
            if attempts <= 2:
                raise ValueError("fail")
            return "done"

        strategy = RetryStrategy(
            max_retries=3,
            backoff=BackoffStrategy.EXPONENTIAL,
            base_delay_seconds=1.0,
        )
        with_retry(fail_twice, strategy, sleep_fn=delays.append)

        assert len(delays) == 2
        assert delays[0] == 1.0  # 1.0 * 2^0
        assert delays[1] == 2.0  # 1.0 * 2^1


class TestFallbackChain:
    """Tests for FallbackChain."""

    def test_first_option_succeeds(self) -> None:
        chain = FallbackChain()
        chain.add_option("model_a", lambda: "result_a")
        chain.add_option("model_b", lambda: "result_b")

        name, result = chain.execute()
        assert name == "model_a"
        assert result == "result_a"

    def test_fallback_to_second_option(self) -> None:
        chain = FallbackChain()
        chain.add_option("model_a", lambda: (_ for _ in ()).throw(RuntimeError("fail_a")))
        chain.add_option("model_b", lambda: "result_b")

        name, result = chain.execute()
        assert name == "model_b"
        assert result == "result_b"

    def test_all_options_fail(self) -> None:
        chain = FallbackChain()
        chain.add_option("a", lambda: (_ for _ in ()).throw(ValueError("fail_a")))
        chain.add_option("b", lambda: (_ for _ in ()).throw(ValueError("fail_b")))

        with pytest.raises(RuntimeError, match="All fallback options failed"):
            chain.execute()

    def test_empty_chain(self) -> None:
        chain = FallbackChain()
        with pytest.raises(RuntimeError, match="no options configured"):
            chain.execute()

    def test_chaining(self) -> None:
        chain = FallbackChain().add_option("a", lambda: "a").add_option("b", lambda: "b")
        assert len(chain.options) == 2


class TestCircuitBreaker:
    """Tests for CircuitBreaker."""

    def test_closed_state_allows_calls(self) -> None:
        cb = CircuitBreaker(failure_threshold=3)
        result = cb.call(lambda: 42)
        assert result == 42
        assert cb.state == CircuitBreaker.State.CLOSED

    def test_opens_after_threshold(self) -> None:
        cb = CircuitBreaker(failure_threshold=3)

        for _ in range(3):
            with pytest.raises(RuntimeError):
                cb.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))

        assert cb.state == CircuitBreaker.State.OPEN
        assert cb.failure_count == 3

    def test_open_state_rejects_calls(self) -> None:
        cb = CircuitBreaker(failure_threshold=1)

        with pytest.raises(RuntimeError):
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))

        with pytest.raises(RuntimeError, match="Circuit breaker is OPEN"):
            cb.call(lambda: 42)

    def test_half_open_after_recovery_timeout(self) -> None:
        current_time = 0.0

        def time_fn() -> float:
            return current_time

        cb = CircuitBreaker(
            failure_threshold=1,
            recovery_timeout_seconds=10.0,
            time_fn=time_fn,
        )

        # Trip the circuit
        with pytest.raises(RuntimeError):
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))

        assert cb.state == CircuitBreaker.State.OPEN

        # Advance time past recovery timeout
        current_time = 15.0
        assert cb.state == CircuitBreaker.State.HALF_OPEN

    def test_reset(self) -> None:
        cb = CircuitBreaker(failure_threshold=1)

        with pytest.raises(RuntimeError):
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))

        assert cb.state == CircuitBreaker.State.OPEN

        cb.reset()
        assert cb.state == CircuitBreaker.State.CLOSED
        assert cb.failure_count == 0

    def test_success_resets_failure_count(self) -> None:
        cb = CircuitBreaker(failure_threshold=3)

        # Two failures
        for _ in range(2):
            with pytest.raises(RuntimeError):
                cb.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))
        assert cb.failure_count == 2

        # One success resets count
        cb.call(lambda: "ok")
        assert cb.failure_count == 0
        assert cb.state == CircuitBreaker.State.CLOSED
