import pytest

from indicspec.harness.gpu_guard import VramState, in_run_ok, max_safe_fraction, preflight


def test_preflight_accepts_fitting_budget():
    s = VramState(total_mib=24463, used_mib=1604)
    preflight(0.85, s)  # 20793 + 512 <= 22859


def test_preflight_rejects_overcommit():
    s = VramState(total_mib=24463, used_mib=1604)
    with pytest.raises(RuntimeError, match="pre-flight"):
        preflight(0.95, s)  # 23239 + 512 > 22859


def test_max_safe_fraction_is_consistent_with_preflight():
    s = VramState(total_mib=24463, used_mib=1604)
    f = max_safe_fraction(s)
    preflight(f - 1e-4, s)
    with pytest.raises(RuntimeError):
        preflight(f + 1e-3, s)


def test_in_run_ceiling():
    assert in_run_ok(22000, 24463)
    assert not in_run_ok(24000, 24463)
