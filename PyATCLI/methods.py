"""
methods.py
----------
The methods paragraph of a paper, written from what was actually done: `methods_text(state)` states the values of every pipeline option in force, each step in
its own words, then the software and package versions.

    methods_text("D:/rats/Rat7")                       # a recording: what is saved for it (its options, its splices), and what it has
    methods_text("Event PETH_L1P.provenance.json")     # a provenance record: what produced that export, with the software that did
    methods_text({"denoise.cutoff_hz": 8})             # a set of options on a typical TDT recording with a 415 nm reference
    methods_text(PipelineOptions(...))                 # the same

    "dF/F was computed by OLS regression of the 415 nm channel onto the 465 nm channel, with the fitted 415 subtracted, then divided by a double-exponential
    photobleaching baseline fitted to the 465 nm channel, then low-pass filtered (Butterworth, 2nd order, 5 Hz, zero-phase). For event-locked analyses the dF/F
    was smoothed with a 0.5 s moving average. Each trial was z-scored against its own baseline, ... Analyses were run with PyATCLI 2026.11.5 ..."

The words are not here: each step of the pipeline registers its own sentence next to its code (pipeline.sentence), so an option added to a step, or a step added to
the pipeline, appears in this paragraph without this module being touched. What this module does is know what a "state" is and put the sentences together: the steps
that make the dF/F (motion correction, the baseline, the filter) are clauses of one sentence, in the order of the pipeline; the others are sentences of their own;
the software sentence ends it.
"""

import os

from . import pipeline, provenance
from .pipeline import PipelineOptions

# What the software sentence names besides the library and Python: the numerical packages that can change a number (all installed ones, in this order).
SOFTWARE_PACKAGES = ("numpy", "scipy", "pandas", "scikit-learn", "tdt")


def _is_record(state):
    return isinstance(state, dict) and "schema" in state and "kind" in state


def facts_of(state=None, *, regression_method=None, splices=None, app=None):
    """What the paragraph is built from, as a dict: {"pipeline": every option's value, "source_type", "regression_method", "splices", "signal_labels", "software"}.
    `state` is one of: None (the defaults); a PipelineOptions or a dict of option names and values; a recording's path (what is saved for it is read, and the recording
    opened to see what it has); a provenance record, or the path of one (what produced that export, with the software that did); a recording state (provenance.
    recording_state); a recording's cache. `regression_method` and `splices` fill in what a cache or a path does not hold; `app` ({"name", "version"}) names the program
    the numbers were made in, when the software is the one running now."""
    software = None
    if isinstance(state, (str, os.PathLike)):
        path = os.fspath(state)
        if os.path.isfile(path) and path.lower().endswith(".json"):
            state = provenance.read_record(path)
        else:
            from .recording import load_recording
            loaded = load_recording(path, regression_method=regression_method or "ols")
            cache = loaded["cache"]
            method = (regression_method or "ols") if cache.get("source") == "TDT" else None
            state = provenance.recording_state(cache, method, loaded["splices"])
    if _is_record(state):
        software = state.get("software")
        recorded = dict(state.get("recording") or {})
        recorded.setdefault("pipeline", (state.get("pipeline") or {}).get("options"))
        state = recorded
    if isinstance(state, PipelineOptions):
        state = {"pipeline": state.to_dict()}
    if state is None:
        state = {}
    if isinstance(state, dict) and "x" in state and "fs" in state:                      # a cache
        state = provenance.recording_state(state, regression_method, splices or [])
    elif isinstance(state, dict) and state and not ({"source_type", "pipeline", "splices", "regression_method", "signal_labels"} & set(state)) \
            and all("." in str(k) for k in state):                                      # {option name: value}
        state = {"pipeline": state}
    facts = dict(state)
    facts["pipeline"] = PipelineOptions(facts.get("pipeline") or {}).to_dict()
    facts["regression_method"] = facts.get("regression_method") or regression_method
    if splices is not None:
        facts["splices"] = splices
    facts["software"] = software or provenance.software_versions(app)
    return facts


def software_sentence(software, regression_method=None, source_type=None):
    """"Analyses were run with PyATCLI 2026.11.5 (in PyAT 4.1.1) on Python 3.14.0, numpy 2.4.3, scipy 1.16.2, ...": the versions in `software`
    (provenance.software_versions), and only the packages that matter to the recording."""
    packages = software.get("packages") or {}
    named = [f"{name} {packages[name]}" for name in SOFTWARE_PACKAGES if name in packages
             and (name != "scikit-learn" or regression_method in ("huber", "ransac")) and (name != "tdt" or source_type in (None, "TDT"))]
    app = software.get("app") or {}
    program = f"PyATCLI {software['library']}" if software.get("library") else "PyATCLI"
    if app.get("name") and app.get("name") != "PyATCLI engine" and app.get("version"):
        program = f"{app['name']} {app['version']} (library: {program})"
    elif app.get("name") and app.get("name") != "PyATCLI engine":
        program = f"{app['name']} ({program})"
    python = f"Python {software['python']}" if software.get("python") else "Python"
    return f"Analyses were run with {program} on {', '.join([python, *named])}"


def processing_text(options=None, regression_method="ols"):
    """How the dF/F was made, as the sentence of a methods section: "dF/F was computed by OLS regression of the 415 nm channel onto the 465 nm channel, ..." — the
    steps that make it (the motion correction, the baseline, the filter) for the pipeline `options` ({option name: value}, a PipelineOptions or None for the
    defaults) of a TDT recording with a 415 nm reference. No trailing full stop; "" if no step has anything to say."""
    facts = facts_of(PipelineOptions.from_dict(options).to_dict() if options is not None else None, regression_method=regression_method)
    resolved = PipelineOptions(facts["pipeline"])
    clauses = []
    for step in pipeline.steps():
        text = step.sentence(resolved, facts) if step.clause and step.sentence is not None else None
        if text:
            clauses.append(text.strip().rstrip("."))
    return ", ".join(clauses)


def methods_text(state=None, *, software=True, regression_method=None, splices=None, app=None):
    """The methods paragraph for `state` (see facts_of for what it can be): every pipeline step's sentence for the values in force, in the order of the pipeline,
    then — unless `software` is False — the software and package versions. A step that has nothing to say for this recording (a TDT step for an Oxysoft export,
    the splices of an unspliced recording) says nothing."""
    facts = facts_of(state, regression_method=regression_method, splices=splices, app=app)
    options = PipelineOptions(facts["pipeline"])
    clauses, sentences = [], []
    for step in pipeline.steps():
        text = step.sentence(options, facts) if step.sentence is not None else None
        if text:
            (clauses if step.clause else sentences).append(text.strip().rstrip("."))
    parts = ([", ".join(clauses)] if clauses else []) + sentences
    if software:
        parts.append(software_sentence(facts["software"], facts.get("regression_method"), facts.get("source_type")))
    return ". ".join(parts) + "." if parts else ""
