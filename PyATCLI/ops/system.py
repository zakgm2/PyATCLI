"""ops/system.py — what is installed, and what can be asked."""

from .. import _optional, pipeline, provenance
from ..analyses import GROUP_KIND, ParameterError, describe_kinds
from . import PROTOCOL_VERSION, describe, operation


@operation("info", "versions of this library, Python and the numerical packages — what a result would be stamped with", result=["python", "implementation", "bits", "platform", "frozen", "library", "packages", "optional", "app", "protocol"])
def info(ctx):
    versions = provenance.software_versions(ctx.app)
    versions["optional"] = _optional.available()          # {extra: everything it needs can be imported}
    versions["protocol"] = PROTOCOL_VERSION
    return versions


@operation("kinds", "the analyses and every parameter of each", result=["analyses"])
def kinds(ctx, *, kind=None):
    analyses = describe_kinds()
    if kind is not None:
        if kind not in analyses:
            raise ParameterError(f"unknown analysis {kind!r}; the analyses are: {', '.join(analyses)}")
        analyses = {kind: analyses[kind]}
    result = {"analyses": analyses}
    if kind is None:
        result["group_analysis"] = {"kind": GROUP_KIND,
                                    "summary": "the same event-locked responses across recordings: `group run SETTINGS` (see `group --help`)"}
    return result


@operation("pipeline.options", "the pipeline's options: every step, each option with its default, what values it takes and what it does (pipeline.py)", result=["steps", "defaults"])
def pipeline_options(ctx):
    return pipeline.describe()


@operation("methods.text", "the methods paragraph for a paper, written from what was done: the pipeline options in force for a recording (its saved choices, splices and "
                           "reference stream), each step in words, and the software versions", result=["text", "options"])
def methods_text_(ctx, *, recording=None, load=None, software=True, app=None):
    from ..methods import methods_text
    from ..provenance import recording_state
    from .common import open_recording

    if recording is None:
        text = methods_text(None, software=bool(software), app=app)
        return {"text": text, "options": pipeline.defaults()}
    loaded, _ = open_recording(ctx, recording, load)
    cache = loaded["cache"]
    if isinstance(recording, dict):
        method = ctx.store.get(recording["id"]).method
    else:
        method = (load or {}).get("regression_method") or "ols"
    state = recording_state(cache, method if cache.get("source") == "TDT" else None, loaded["splices"])
    return {"text": methods_text(state, software=bool(software), app=app), "options": dict(cache.get("pipeline") or pipeline.defaults())}


@operation("operations", "every operation the engine can be asked for, with its parameters", result=["protocol", "operations"])
def operations(ctx):
    return {"protocol": PROTOCOL_VERSION, "operations": describe()}
