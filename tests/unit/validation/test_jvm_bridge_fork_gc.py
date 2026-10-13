"""A fork child of a process whose in-process JVM is running must not hang in garbage collection.

JPype registers ``_jpype._collect`` in ``gc.callbacks`` when it starts the JVM. After a
``fork`` the JVM does not exist in the child (only the forking thread survives), but the
callback is still registered: the child's first garbage collection calls ``System.gc``
into a JVM whose VM thread is gone and waits forever (audit 2026-10-13,
`internal notes`, defect 2). ``jvm_bridge`` removes the
callback in every forked child.

The probe runs in a subprocess so a regression can only hang (and be killed) there.

The check is deterministic by testing the mechanism: whether a collection reaches the dead
VM thread depends on JPype's own GC heuristics, so a bare ``gc.collect`` in the child hangs
only some of the time on the unfixed code. The parent therefore asserts the callback is
registered (so the test cannot pass for nothing if JPype stops registering it), and the child
exits 3 if the callback is still registered after the fork. The ``gc.collect`` and the
30 s watchdog stay as the behavioural backstop.
"""
from __future__ import annotations

import subprocess
import sys

from tests.support.jars import jar_or_skip

_PROBE = r"""
import gc, os, signal, time
import _jpype
from orthonym import jvm_bridge
assert jvm_bridge.opsin_available(), "in-process JVM did not start"
jvm_bridge.opsin_stdout("ethanol", False)       # the JVM is up, as in a naming worker
assert _jpype._collect in gc.callbacks, "JPype no longer registers _collect: the probe tests nothing"
pid = os.fork()
if pid == 0:
    if _jpype._collect in gc.callbacks:         # the mechanism itself: the dead VM's callback
        os._exit(3)
    gc.collect()                                # behavioural backstop: must not hang
    os._exit(0)
deadline = time.time() + 30
while time.time() < deadline:
    done, status = os.waitpid(pid, os.WNOHANG)
    if done:
        print("CHILD", os.waitstatus_to_exitcode(status), flush=True)
        break
    time.sleep(0.1)
else:
    os.kill(pid, signal.SIGKILL)
    os.waitpid(pid, 0)
    print("CHILD HUNG", flush=True)
"""

# A child that ends through sys.exit runs JPype's atexit shutdown, which would call into the
# dead JVM unless the after-fork hook cleared jpype.config.onexit.
_PROBE_SYS_EXIT = _PROBE.replace(
    """    if _jpype._collect in gc.callbacks:         # the mechanism itself: the dead VM's callback
        os._exit(3)
    gc.collect()                                # behavioural backstop: must not hang
    os._exit(0)""",
    """    import sys, jpype.config
    if jpype.config.onexit:                     # the mechanism itself: atexit would shut down the dead VM
        os._exit(4)
    sys.exit(0)                                 # behavioural backstop: atexit must not hang""")
assert _PROBE_SYS_EXIT != _PROBE


def test_forked_child_of_a_started_jvm_survives_its_first_gc():
    jar_or_skip()  # the probe needs the in-process JVM
    out = subprocess.run([sys.executable, "-c", _PROBE], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0 and "CHILD 0" in out.stdout, (
        f"rc={out.returncode}\n{out.stdout}\n{out.stderr[-2000:]}")


def test_forked_child_of_a_started_jvm_exits_through_sys_exit():
    jar_or_skip()  # the probe needs the in-process JVM
    out = subprocess.run([sys.executable, "-c", _PROBE_SYS_EXIT], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0 and "CHILD 0" in out.stdout, (
        f"rc={out.returncode}\n{out.stdout}\n{out.stderr[-2000:]}")
