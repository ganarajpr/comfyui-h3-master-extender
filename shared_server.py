"""One model server for a whole job: the Shot Writer leaves it running, the Extender's rewriter attaches to it.

Both nodes start the same model on ninfer-serve (about 10 s of load each time). At most ONE server is held here. The writer ``use``s it with ``keep`` and
it is parked when the writer finishes; the rewriter's ``use`` finds the parked server for the same model, checks that it fits (``fits(meta)``) and attaches
instead of launching. Whoever uses it last closes it, so the render that follows has the card exactly as before.

A parked server nobody claims is closed by: the next ``use`` for anything else, ComfyUI's cancel (``install``), ComfyUI's unload (``vram_release``),
the Extender at its own entry to the render, a timer (``PARK_SECONDS``) and process exit.
"""

import atexit
import contextlib
import logging
import os
import threading

try:
    from . import vram_release
except ImportError:  # imported as a top-level module (tests)
    import vram_release

_LOG = logging.getLogger("minimax_h3_master_extender.shared_server")

PARK_SECONDS = 300.0
_WRAPPED_FLAG = "_h3_master_shared_server"
_LOCK = threading.RLock()
_HANDLE = None


class Handle:
    def __init__(self, key, server, closer, meta):
        self.key = key
        self.server = server
        self.closer = closer
        self.meta = meta
        self.attached = False
        self.parked = False
        self.timer = None


def key_of(model_path: str, mmproj_path: str = "") -> tuple:
    def norm(path):
        return os.path.normcase(os.path.abspath(path)) if path else ""
    return (norm(model_path), norm(mmproj_path))


def current():
    return _HANDLE


def _stop_timer(handle) -> None:
    timer, handle.timer = handle.timer, None
    if timer is not None:
        timer.cancel()


def close(handle=None) -> None:
    """Close the live server (``handle``: only if it is still the live one). Never raises."""
    global _HANDLE
    with _LOCK:
        held = _HANDLE
        if held is None or (handle is not None and handle is not held):
            return
        _HANDLE = None
        _stop_timer(held)
        try:
            held.closer()
        except Exception:
            _LOG.warning("shared server: closing failed", exc_info=True)
        _LOG.info("shared server: closed")


def close_parked() -> None:
    with _LOCK:
        if _HANDLE is not None and _HANDLE.parked:
            _LOG.info("shared server: closing the parked server (nobody claimed it)")
            close(_HANDLE)


def park(handle) -> None:
    with _LOCK:
        if handle is not _HANDLE:
            return
        handle.parked = True
        _stop_timer(handle)
        timer = threading.Timer(PARK_SECONDS, close_parked)
        timer.daemon = True
        handle.timer = timer
        timer.start()
    _LOG.info("shared server: kept for the Extender that reads this node's clips")


def acquire(key, launch, *, fits=None, say=None):
    """The live handle for ``key``: the parked server when it fits (attached), else a new one from ``launch() -> (server, closer, meta)``.

    A parked or running server of another key, or one that does not fit, is closed first. ``handle.server`` is None when ``launch`` found no server."""
    global _HANDLE
    say = say or (lambda message: None)
    with _LOCK:
        held = _HANDLE
        if held is not None:
            if held.parked and held.key == key and (fits is None or fits(held.meta)):
                held.parked = False
                _stop_timer(held)
                held.attached = True
                say("attached to the model server the Shot Writer left running (no second start)")
                return held
            _LOG.info("shared server: closing the held server (%s)", "another model or settings" if held.parked else "still in use")
            close(held)
        server, closer, meta = launch()
        if server is None:
            closer()
            return Handle(key, None, lambda: None, meta)
        _HANDLE = Handle(key, server, closer, meta)
        return _HANDLE


@contextlib.contextmanager
def use(key, launch, *, fits=None, say=None, keep=None):
    """``acquire`` for a ``with`` block. On a normal exit the server is parked when ``keep()`` is true, else closed; on any error or cancel it is closed."""
    handle = acquire(key, launch, fits=fits, say=say)
    done = False
    try:
        yield handle.server
        done = True
    finally:
        if done and handle.server is not None and keep is not None and keep():
            park(handle)
        else:
            close(handle)


def install(model_management=None) -> bool:
    """Make ComfyUI's cancel close a parked server. Returns True when wrapped now."""
    if model_management is None:
        try:
            import comfy.model_management as model_management
        except Exception:
            return False
    original = getattr(model_management, "interrupt_current_processing", None)
    if original is None or getattr(original, _WRAPPED_FLAG, False):
        return False

    def interrupt_current_processing(*args, **kwargs):
        result = original(*args, **kwargs)
        if kwargs.get("value", args[0] if args else True):
            close_parked()
        return result

    setattr(interrupt_current_processing, _WRAPPED_FLAG, True)
    interrupt_current_processing.__wrapped__ = original
    model_management.interrupt_current_processing = interrupt_current_processing
    return True


vram_release.register(close_parked)
atexit.register(close)
