"""Single-instance mutex: second acquirer loses, release frees the name.

Uses a test-only mutex name so the tests pass while the real app is
running (which holds the production mutex).
"""

import ctypes

import pytest

from app import single_instance


@pytest.fixture(autouse=True)
def _test_mutex_name(monkeypatch):
    monkeypatch.setattr(
        single_instance, "_MUTEX_NAME", "Local\\SpeakEasy.Test.SingleInstance"
    )


def _second_holder_fails() -> bool:
    """Simulate a second process: create the same named mutex directly."""
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.CreateMutexW(None, False, single_instance._MUTEX_NAME)
    already = kernel32.GetLastError() == single_instance._ERROR_ALREADY_EXISTS
    if handle:
        kernel32.CloseHandle(handle)
    return already


def test_first_acquire_wins_second_loses():
    try:
        assert single_instance.acquire()
        assert single_instance.acquire()  # idempotent within the process
        assert _second_holder_fails()
    finally:
        single_instance.release()


def test_release_frees_the_name():
    assert single_instance.acquire()
    single_instance.release()
    assert not _second_holder_fails()
    # Clean up the probe's implicit slot by re-acquiring and releasing.
    assert single_instance.acquire()
    single_instance.release()
