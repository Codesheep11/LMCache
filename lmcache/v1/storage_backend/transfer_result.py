# SPDX-License-Identifier: Apache-2.0
"""Structured per-batch results for bounded xDS transfers."""

# Standard
from dataclasses import dataclass


@dataclass(frozen=True)
class BackendTransferResult:
    completed: tuple[bool, ...]
    requested_bytes: int
    completed_bytes: int
    device: str
    window_size: int
    window_count: int
    peak_staging_bytes: int
    storage_seconds: float
    gpu_seconds: float
    overlap_seconds: float
    bank_wait_seconds: float
    short_io_chunks: int
    timeout_chunks: int

    def __post_init__(self) -> None:
        if self.requested_bytes < 0 or self.completed_bytes < 0:
            raise ValueError("transfer byte counts must be non-negative")
        if self.completed_bytes > self.requested_bytes:
            raise ValueError("completed bytes exceed requested bytes")
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")
        if self.window_count < 0:
            raise ValueError("window_count must be non-negative")
        if self.peak_staging_bytes < 0:
            raise ValueError("peak_staging_bytes must be non-negative")
        if min(
            self.storage_seconds,
            self.gpu_seconds,
            self.overlap_seconds,
            self.bank_wait_seconds,
        ) < 0:
            raise ValueError("transfer durations must be non-negative")
        if self.short_io_chunks < 0 or self.timeout_chunks < 0:
            raise ValueError("transfer error counts must be non-negative")

    @property
    def failed_chunks(self) -> int:
        return len(self.completed) - sum(self.completed)
