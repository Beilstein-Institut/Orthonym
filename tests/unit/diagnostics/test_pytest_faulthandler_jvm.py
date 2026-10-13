"""The test session must not take SIGSEGV away from the in-process JVM.

Audit 2026-10-13 (internal notes section 1, "Related"):
pytest's own ``faulthandler`` plugin calls ``faulthandler.enable`` at ``pytest_configure``,
before the first test starts the JPype JVM, and ``faulthandler.disable`` at
``pytest_unconfigure``, while the JVM still runs. ``disable`` restores the disposition saved
at ``enable``, SIG_DFL, over HotSpot's SIGSEGV handler. The JVM raises SIGSEGV itself in normal
operation (an implicit null check, a safepoint poll), so the next such fault during worker
teardown kills the worker (death by signal 11, no hs_err, no traceback), and ``pytest-xdist``
does not see it because no test is running. ``tests/conftest.py`` removes the plugin's
``unconfigure`` half after its ``configure`` ran, which keeps faulthandler on for the whole
session and never disables it.

Both checks run a complete pytest session in a child process (the sequence under test is
pytest's own configure and unconfigure), start the JVM in it, and then raise Java
NullPointerExceptions from a hook that runs after pytest's faulthandler plugin has unconfigured.
The control reproduces the plugin's sequence by hand and must die: it shows the probe can fail.
"""
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from tests.support.jars import jar_or_skip

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Run as `python -c`. argv[1] is "session" (what the suite does) or "control" (what the plugin
# does, written out). Prints one line per fact; the last line of a surviving run is SURVIVED.
_PROBE = r"""
import faulthandler, sys
import pytest

MODE = sys.argv[1]

class Probe:
    def pytest_configure(self, config):
        if MODE == "control":
            faulthandler.enable()       # before the JVM exists, as the plugin does at configure

    def pytest_sessionstart(self, session):
        import jpype
        from orthonym import jvm_bridge
        assert jvm_bridge.opsin_available(), "in-process JVM did not start"
        assert jpype.isJVMStarted()

    # trylast: after the faulthandler plugin's own pytest_unconfigure
    @pytest.hookimpl(trylast=True)
    def pytest_unconfigure(self, config):
        import jpype
        if MODE == "control":
            faulthandler.disable()      # the plugin's unconfigure, with the JVM running
        print("FAULTHANDLER_ON", faulthandler.is_enabled(), flush=True)
        JString = jpype.JClass("java.lang.String")
        CharArr = jpype.JArray(jpype.JChar)
        npes = 0
        for i in range(3000):           # a Java NPE raised in the JVM's own code is a SIGSEGV
            try:
                JString.valueOf(None if i % 2 else CharArr(3))
            except Exception:
                npes += 1
        print("SURVIVED", npes, flush=True)

rc = pytest.main(
    ["--collect-only", "-q", "-p", "no:cacheprovider", "-o", "addopts=",
     "tests/unit/diagnostics/test_crash_traceback.py"],
    plugins=[Probe()],
)
sys.exit(int(rc))
"""


def _session(mode: str):
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTEST_XDIST")}
    return subprocess.run([sys.executable, "-c", _PROBE, mode], capture_output=True, text=True,
                          timeout=300, cwd=PROJECT_ROOT, env=env)


def test_a_test_session_leaves_hotspots_sigsegv_handler_in_place():
    """After pytest has unconfigured its faulthandler plugin, the JVM survives Java NPEs, and
    faulthandler is still on (a native crash in RDKit keeps its Python traceback)."""
    jar_or_skip()
    out = _session("session")
    assert out.returncode == 0 and "SURVIVED 1500" in out.stdout, (
        f"rc={out.returncode}\n{out.stdout[-2000:]}\n{out.stderr[-2000:]}")
    assert "FAULTHANDLER_ON True" in out.stdout, out.stdout


@pytest.mark.skipif(sys.platform == "win32", reason="the JVM's SIGSEGV use is POSIX")
def test_control_disabling_faulthandler_under_a_running_jvm_kills_the_process():
    """The pytest plugin's sequence, written out, dies on the first Java NPE: the probe above
    has teeth."""
    jar_or_skip()
    out = _session("control")
    assert out.returncode in (-signal.SIGSEGV, 128 + signal.SIGSEGV), (
        f"rc={out.returncode}\n{out.stdout[-2000:]}\n{out.stderr[-2000:]}")
    assert "SURVIVED" not in out.stdout, out.stdout
