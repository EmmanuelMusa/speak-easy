"""Windows power state: battery detection + power-throttling opt-out.

On battery, Windows 11 applies Power Throttling (EcoQoS) to background
processes: efficiency-core scheduling, low CPU clocks, coalesced timers.
For a dictation pipeline that means slow VAD/Whisper passes, sloppy poll
loops, and in the worst case dropped audio frames that read as "no speech
detected". opt_out_of_power_throttling() tells Windows this process must
run at normal speed regardless of power source.

The NVIDIA GPU still downclocks on battery (driver policy, not reachable
from user code) — that's handled by scaling the Ollama timeout instead
(see cleanup.py + [cleanup] battery_timeout_multiplier).
"""

from __future__ import annotations

import ctypes
import logging
import sys

log = logging.getLogger(__name__)

# winnt.h / processthreadsapi.h
_PROCESS_POWER_THROTTLING_CURRENT_VERSION = 1
_PROCESS_POWER_THROTTLING_EXECUTION_SPEED = 0x1      # EcoQoS
_PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION = 0x4
_ProcessPowerThrottling = 4  # PROCESS_INFORMATION_CLASS


class _PROCESS_POWER_THROTTLING_STATE(ctypes.Structure):
    _fields_ = [
        ("Version", ctypes.c_ulong),
        ("ControlMask", ctypes.c_ulong),
        ("StateMask", ctypes.c_ulong),
    ]


class _SYSTEM_POWER_STATUS(ctypes.Structure):
    _fields_ = [
        ("ACLineStatus", ctypes.c_byte),      # 0 battery, 1 AC, -1 unknown
        ("BatteryFlag", ctypes.c_byte),
        ("BatteryLifePercent", ctypes.c_byte),
        ("SystemStatusFlag", ctypes.c_byte),
        ("BatteryLifeTime", ctypes.c_ulong),
        ("BatteryFullLifeTime", ctypes.c_ulong),
    ]


def opt_out_of_power_throttling() -> bool:
    """Exempt this process from EcoQoS + timer coalescing. True on success."""
    if sys.platform != "win32":
        return False
    state = _PROCESS_POWER_THROTTLING_STATE(
        Version=_PROCESS_POWER_THROTTLING_CURRENT_VERSION,
        # Take control of both mechanisms; StateMask=0 disables them.
        ControlMask=(
            _PROCESS_POWER_THROTTLING_EXECUTION_SPEED
            | _PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION
        ),
        StateMask=0,
    )
    kernel32 = ctypes.windll.kernel32
    # Explicit prototypes: HANDLE is 64-bit; without these ctypes passes the
    # GetCurrentProcess pseudo-handle as a truncated 32-bit int and the call
    # fails with ERROR_INVALID_HANDLE.
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.SetProcessInformation.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_ulong,
    ]
    ok = kernel32.SetProcessInformation(
        kernel32.GetCurrentProcess(),
        _ProcessPowerThrottling,
        ctypes.byref(state),
        ctypes.sizeof(state),
    )
    if ok:
        log.info("Power throttling opt-out applied (full speed on battery)")
    else:
        log.warning(
            "Power throttling opt-out failed (error %d); expect slowdown "
            "on battery", kernel32.GetLastError(),
        )
    return bool(ok)


def on_battery() -> bool:
    """True when running on battery (AC unplugged). False if unknown."""
    if sys.platform != "win32":
        return False
    status = _SYSTEM_POWER_STATUS()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):
        return False
    return status.ACLineStatus == 0
