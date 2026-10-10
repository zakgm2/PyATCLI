"""
engine/contract.py
------------------
What a program that talks to the engine (`python -m PyATCLI serve`) can count on, written down so a change that would break it is noticed:

    build()                        the contract as it is now: the protocol version, every operation with its parameters, and every analysis with
                                   its parameters and the tables it returns with the columns that are always there, and the tables written
                                   besides ("outputs": a group analysis's folder with the file of each table, an Export View's data). ENGINE_CONTRACT.json in the
                                   repository is this, generated (`python -m PyATCLI docs --format contract --out ENGINE_CONTRACT.json`)
    breaking_changes(old, new)     what `new` no longer offers of what `old` did: a caller written against `old` would break

Adding is never breaking: a new operation, an optional parameter, a new table or column. These are:

    an operation, or one of its parameters, that is gone; a parameter that must now be given (it was optional, or did not exist)
    a key of an operation's result that was always there and no longer is; a key of the loaded recording (`recording.CACHE_KEYS`) likewise
    an analysis, one of its parameters, or a choice of one, that is gone; a parameter that must now be given or has another type
    a table of an analysis that is gone, or a column that is always there and no longer is; likewise for the outputs, and a file of a group's
    folder that is gone or has another name
    a protocol version that went down

A breaking change means a new `ops.PROTOCOL_VERSION`. tests/contracts/protocol_N.json is the contract as it was when protocol N began, and
tests/test_engine_contract.py checks that the engine still honours the one of its own protocol: change something it lists and the test fails
until the protocol is raised (and the new baseline added), which is also what tells a window running an older engine to say so (the app
compares the protocol in the engine's `ready` message with the ones it speaks).
"""

from .. import ops
from ..analyses import describe_kinds, describe_outputs
from ..recording import CACHE_KEYS


def build():
    """The contract as the code has it now (JSON-ready)."""
    operations = {name: {"summary": op["summary"], "parameters": [dict(p) for p in op["parameters"]], "result": list(op["result"])}
                  for name, op in ops.describe().items()}
    analyses = {kind: {"summary": a["summary"],
                       "parameters": [{k: p[k] for k in ("name", "type", "required", "default", "choices")} for p in a["parameters"]],
                       "tables": {name: list(cols) for name, cols in a["tables"].items()}}
                for kind, a in describe_kinds().items()}
    outputs = {name: {"summary": o["summary"], "tables": {t: list(cols) for t, cols in o["tables"].items()}, "files": dict(o["files"])}
               for name, o in describe_outputs().items()}
    return {"protocol": ops.PROTOCOL_VERSION, "operations": operations, "analyses": analyses, "outputs": outputs,
            "recording_cache": {source: list(keys) for source, keys in CACHE_KEYS.items()}}


def _params(items):
    return {p["name"]: p for p in items}


def _newly_required(old_p, new_p):
    return bool(new_p.get("required")) and not (old_p is not None and old_p.get("required"))


def breaking_changes(old, new):
    """[a sentence for each thing `old` offered that `new` does not] — empty if a caller written against `old` still works against `new`."""
    problems = []
    if new.get("protocol", 0) < old.get("protocol", 0):
        problems.append(f"the protocol went down, from {old['protocol']} to {new['protocol']}")
    for name, op in old.get("operations", {}).items():
        now = new.get("operations", {}).get(name)
        if now is None:
            problems.append(f"operation {name!r} is gone")
            continue
        before, after = _params(op["parameters"]), _params(now["parameters"])
        for p in before:
            if p not in after:
                problems.append(f"operation {name!r}: parameter {p!r} is gone")
        for p, spec in after.items():
            if _newly_required(before.get(p), spec):
                problems.append(f"operation {name!r}: parameter {p!r} must now be given")
        for key in op.get("result", []):
            if key not in now.get("result", []):
                problems.append(f"operation {name!r}: its result no longer always has {key!r}")
    for kind, a in old.get("analyses", {}).items():
        now = new.get("analyses", {}).get(kind)
        if now is None:
            problems.append(f"analysis {kind!r} is gone")
            continue
        before, after = _params(a["parameters"]), _params(now["parameters"])
        for p, spec in before.items():
            if p not in after:
                problems.append(f"analysis {kind!r}: parameter {p!r} is gone")
                continue
            if after[p].get("type") != spec.get("type"):
                problems.append(f"analysis {kind!r}: parameter {p!r} changed type, from {spec.get('type')} to {after[p].get('type')}")
            for choice in spec.get("choices") or []:
                if choice not in (after[p].get("choices") or []):
                    problems.append(f"analysis {kind!r}: parameter {p!r} no longer accepts {choice!r}")
        for p, spec in after.items():
            if _newly_required(before.get(p), spec):
                problems.append(f"analysis {kind!r}: parameter {p!r} must now be given")
        for table, columns in a.get("tables", {}).items():
            have = now.get("tables", {}).get(table)
            if have is None:
                problems.append(f"analysis {kind!r}: table {table!r} is gone")
                continue
            for column in columns:
                if column not in have:
                    problems.append(f"analysis {kind!r}: table {table!r} has no column {column!r} any more")
    for name, o in old.get("outputs", {}).items():
        now = new.get("outputs", {}).get(name)
        if now is None:
            problems.append(f"output {name!r} is gone")
            continue
        for table, columns in o.get("tables", {}).items():
            have = now.get("tables", {}).get(table)
            if have is None:
                problems.append(f"output {name!r}: table {table!r} is gone")
                continue
            for column in columns:
                if column not in have:
                    problems.append(f"output {name!r}: table {table!r} has no column {column!r} any more")
        for role, filename in o.get("files", {}).items():
            if now.get("files", {}).get(role) != filename:
                problems.append(f"output {name!r}: {role!r} is no longer written to {filename!r}")
    for source, keys in old.get("recording_cache", {}).items():
        have = new.get("recording_cache", {}).get(source)
        if have is None:
            problems.append(f"the loaded recording: no keys are promised for {source!r} any more")
            continue
        for key in keys:
            if key not in have:
                problems.append(f"the loaded recording ({source}) no longer always has {key!r}")
    return problems
