"""Reload modules for the length of one test, then put them back.

A test that reloads an orthonym module (for example so that the module's
import-time read of an environment variable sees the test's environment) changes
that module for the rest of the pytest worker. ``importlib.reload`` re-executes a
module inside the SAME module object, so the reloaded constants and the new class
objects stay there after the test; ``monkeypatch`` restoring the environment does
not undo them. Every later test on that worker then runs with the reloaded
constants and sees other class objects, so ``isinstance`` checks against the class
it imported fail.

``reloaded`` keeps the reload for the body of the test and afterwards puts each
module's original namespace back into the module object, so the rest of the run
sees exactly the objects it imported: the default constants and the original
classes and functions. The functions defined in the module read their globals from
that same namespace, so they see the restored values too. ``tests/conftest.py``
errors any test that reloads an orthonym module and leaves it changed.
"""
from __future__ import annotations

import contextlib
import importlib
from types import ModuleType
from typing import Iterator


@contextlib.contextmanager
def reloaded(*modules: ModuleType) -> Iterator[None]:
    """Reload ``modules`` in order for the body; restore their namespaces on exit.

    Set the environment the reload should read (``monkeypatch.setenv``) before
    entering. The namespaces are restored in reverse order even when the reload or
    the body raises.
    """
    saved = [(module, dict(vars(module))) for module in modules]
    try:
        for module in modules:
            importlib.reload(module)
        yield
    finally:
        for module, namespace in reversed(saved):
            live = vars(module)
            live.clear()
            live.update(namespace)
