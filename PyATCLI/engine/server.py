"""
engine/server.py
----------------
`python -m PyATCLI serve`: the engine as a long-lived process that a window talks to over its standard input and output, one JSON
object per line, so the window never imports the numerical code and never freezes while it runs.

    window -> engine   {"id": 7, "method": "recording.open", "params": {"path": "D:/rats/Rat7"}}
                       {"cancel": 7}                      stop request 7 (at its next progress report, or before it starts)
                       {"method": "shutdown"}             finish and exit (so does closing standard input)
    engine -> window   {"event": "ready", "protocol": 1, "library": "...", "pid": 1234, "data_dir": "...", "operations": [...]}
                       {"id": 7, "event": "progress", "fraction": 0.4, "message": "Reading and processing"}
                       {"id": 7, "event": "warning" | "note" | "state", ...}
                       {"id": 7, "event": "result", "ok": true, "result": {...}}
                       {"id": 7, "event": "error", "ok": false, "error": {"code", "message", "hint", "type"}}
                       {"event": "bye"}

Requests are done one at a time, in order; a second thread listens for `cancel` and `shutdown` while one runs. The methods are the operations
(ops/): `operations` lists them with their parameters. Bulk data — a recording's arrays, a big table — is written to files under `data_dir`
and named in the result, never sent through the pipe (see wire.py). Everything on standard output is protocol; anything a library prints on
its own is sent to standard error, where `--verbose` shows it.
"""

import argparse
import json
import os
import queue
import sys
import threading
import traceback

from .. import ops, provenance
from ..ops import error_info, is_expected, to_jsonable
from ..progress import Cancelled
from .session import Session


class _Discard:
    def write(self, text):
        return len(text)

    def flush(self):
        pass


class _Gone(BaseException):
    """The window's end of the pipe is closed: there is nobody left to answer. (A BaseException, like progress.Cancelled: an analysis that
    is told its progress cannot be heard stops at the next report instead of computing for a window that is no longer there.)"""


def serve(stdin, stdout, stderr, data_dir=None, app=None, verbose=False):
    """Runs the engine on the given streams until it is asked to stop, its input closes, or its output does (the window died: the engine
    ends quietly, with no traceback — a frozen windowed program would show one in a dialog nobody can see). Returns 0."""
    session = Session(data_dir=data_dir, app=app)
    lock = threading.Lock()

    def send(obj):
        line = json.dumps(to_jsonable(obj), ensure_ascii=True)
        try:
            with lock:
                stdout.write(line + "\n")
                stdout.flush()
        except (OSError, ValueError) as e:                   # a closed pipe is a BrokenPipeError / OSError; a closed file is a ValueError
            raise _Gone() from e

    jobs = queue.Queue()
    cancelled = set()

    def listen():
        try:
            _listen()
        except _Gone:                                        # an answer to a bad line could not be sent: the window is gone
            jobs.put(None)

    def _listen():
        for raw in stdin:
            line = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
            if not line.strip():
                continue
            try:
                message = json.loads(line)
                if not isinstance(message, dict):
                    raise ValueError("a message is a JSON object")
            except ValueError as e:
                send({"id": None, "event": "error", "ok": False, "error": {"code": "invalid_input", "message": f"not a request: {e}", "hint": None, "type": "ValueError"}})
                continue
            if "cancel" in message:
                cancelled.add(message["cancel"])
            elif message.get("method") == "shutdown":
                jobs.put(None)
                return
            else:
                jobs.put(message)
        jobs.put(None)                                       # the window closed its end: finish

    threading.Thread(target=listen, name="engine-listener", daemon=True).start()
    try:
        _serve_requests(session, send, jobs, cancelled, stderr)
    except _Gone:
        pass
    finally:
        session.close()
    return 0


def _serve_requests(session, send, jobs, cancelled, stderr):
    send({"event": "ready", "protocol": ops.PROTOCOL_VERSION, "library": provenance.library_version(), "pid": os.getpid(),
          "data_dir": session.data_dir, "operations": ops.names()})
    while True:
        request = jobs.get()
        if request is None:
            break
        rid = request.get("id")

        def emit(event, _rid=rid, **fields):
            send({"id": _rid, "event": event, **fields})

        try:
            if rid in cancelled:
                raise Cancelled()
            if not isinstance(request.get("method"), str):
                raise ValueError("a request has a method")
            result = session.handle(request["method"], request.get("params"), emit, lambda _rid=rid: _rid in cancelled)
            send({"id": rid, "event": "result", "ok": True, "result": result})
        except Cancelled:
            send({"id": rid, "event": "error", "ok": False, "error": error_info(Cancelled())})
        except Exception as e:                               # noqa: BLE001 — an error in one request is that request's answer, not the engine's end
            if not is_expected(e):
                print(traceback.format_exc(), file=stderr)
            send({"id": rid, "event": "error", "ok": False, "error": error_info(e)})
        finally:
            cancelled.discard(rid)
    send({"event": "bye"})


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m PyATCLI serve", description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog="The protocol is described in PyATCLI/engine/server.py and API_REFERENCE.md (Engine).")
    p.add_argument("--data-dir", metavar="DIR", help="where bulk results (arrays, big tables) are written (default: a temporary folder, removed on exit)")
    p.add_argument("--app-name", default="PyATCLI engine", help="the program asking, for the provenance records the engine writes")
    p.add_argument("--app-version", help="its version")
    p.add_argument("--verbose", "-v", action="store_true", help="show what libraries print, and the tracebacks of bugs, on stderr")
    args = p.parse_args(argv)
    app = {"name": args.app_name}
    if args.app_version:
        app["version"] = args.app_version
    protocol_out = sys.stdout
    try:
        protocol_out.reconfigure(newline="\n", encoding="utf-8")        # one LF per message, whatever the platform's text mode
    except (AttributeError, ValueError):
        pass
    sys.stdout = sys.stderr if args.verbose else _Discard()   # a library that prints must not corrupt the protocol
    try:
        return serve(sys.stdin.buffer, protocol_out, sys.stderr, data_dir=args.data_dir, app=app, verbose=args.verbose)
    except KeyboardInterrupt:
        return 130
    except Exception:                                        # noqa: BLE001 — a bug in the engine: say so on stderr, and end (not a dialog)
        traceback.print_exc(file=sys.stderr)
        return 1
    finally:
        try:
            protocol_out.flush()
        except (OSError, ValueError):                         # the window is gone: Python's own flush at exit must not complain about it
            protocol_out = _Discard()
        sys.stdout = protocol_out


if __name__ == "__main__":
    sys.exit(main())
