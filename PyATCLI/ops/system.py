"""ops/system.py — what is installed, and what can be asked."""

from .. import provenance
from ..analyses import GROUP_KIND, ParameterError, describe_kinds
from . import PROTOCOL_VERSION, describe, operation


@operation("info", "versions of this library, Python and the numerical packages — what a result would be stamped with", result=["python", "implementation", "bits", "platform", "frozen", "library", "packages", "app", "protocol"])
def info(ctx):
    versions = provenance.software_versions(ctx.app)
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


@operation("operations", "every operation the engine can be asked for, with its parameters", result=["protocol", "operations"])
def operations(ctx):
    return {"protocol": PROTOCOL_VERSION, "operations": describe()}
