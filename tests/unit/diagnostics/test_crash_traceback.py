"""R1 (audit 2026-09-03): a native crash (SIGSEGV inside RDKit or the JVM bridge)
must leave a Python traceback, not only a JVM hs_err file. The 72 hs_err logs of
Aug 30 - Sep 2 named no Python line because nothing had enabled faulthandler.

Audit 2026-10-13 (internal notes): faulthandler and the
in-process JVM compete for the SIGSEGV handler. ``faulthandler.disable`` restores the
handler saved before the JVM started, which removes HotSpot's; a later
``faulthandler.enable`` then puts faulthandler's there, and the JVM's next ordinary
SIGSEGV (an implicit null check, a safepoint poll) kills the process with no traceback.
So every check that touches faulthandler's state runs in a subprocess, never in the
shared test worker.
"""
import subprocess
import sys

from tests.support.jars import jar_or_skip


def _run(code, timeout=120):
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=timeout)


def test_enable_crash_traceback_turns_faulthandler_on():
    code = (
        "import faulthandler\n"
        "from orthonym.diagnostics import enable_crash_traceback\n"
        "faulthandler.disable()\n"
        "enable_crash_traceback()\n"
        "assert faulthandler.is_enabled()\n"
        "enable_crash_traceback()  # idempotent\n"
        "assert faulthandler.is_enabled()\n"
        "print('FH ON')\n"
    )
    out = _run(code)
    assert out.returncode == 0 and "FH ON" in out.stdout, out.stdout + out.stderr


def test_cli_enables_faulthandler_before_naming():
    code = (
        "import faulthandler, sys\n"
        "import orthonym.cli as cli\n"
        "faulthandler.disable()\n"
        "rc = cli.main(['CCO'])\n"
        "print('FH', faulthandler.is_enabled(), rc)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert "FH True 0" in out.stdout, out.stdout + out.stderr


# A Java NullPointerException raised from compiled code is a JVM-internal SIGSEGV, which
# HotSpot handles itself. Half of these calls pass null (String.valueOf(char) throws).
_NPE_AFTER_ENABLE = r"""
import jpype
from orthonym import jvm_bridge
assert jvm_bridge.opsin_available(), "in-process JVM did not start"
from orthonym.diagnostics import enable_crash_traceback
enable_crash_traceback()          # after the JVM is up: must leave HotSpot's SIGSEGV handler alone
JString = jpype.JClass("java.lang.String")
CharArr = jpype.JArray(jpype.JChar)
npes = 0
for i in range(5000):
    try:
        JString.valueOf(None if i % 2 else CharArr(3))
    except Exception:
        npes += 1
print("SURVIVED", npes, flush=True)
"""


def test_enable_crash_traceback_after_jvm_start_does_not_take_over_sigsegv():
    """Called after the first naming call, enable_crash_traceback must not replace
    HotSpot's SIGSEGV handler: the JVM's own faults would then kill the process."""
    jar_or_skip()  # the probe needs the in-process JVM
    out = _run(_NPE_AFTER_ENABLE)
    assert out.returncode == 0 and "SURVIVED 2500" in out.stdout, (
        f"rc={out.returncode}\n{out.stdout}\n{out.stderr[-2000:]}")
