"""
pipeline.py
-----------
The processing a recording goes through, as options with defaults, in one registry: the dF/F baseline, the low-pass filter, the window the motion-correction
regression is fitted over, the smoothing every event-locked analysis starts with, and the baseline a z-score is taken against. What the options are, what they
default to, what values are allowed and what each one does to the sentence of the methods paragraph (methods.py) is declared where the step's code is, and nowhere
else:

    pipeline.register_step("denoise", "Low-pass filter", order=30, options=[
        pipeline.Option("denoise", "cutoff_hz", "float", 5.0, "Cutoff", "The cutoff of the low-pass filter.", minimum=0.01, unit="Hz")])

    @pipeline.sentence("denoise")
    def _denoise_sentence(options, state):
        return f"then low-pass filtered (Butterworth, {options['denoise.cutoff_hz']:g} Hz)"

and the code reads the value where it uses it: `options["denoise.cutoff_hz"]`. A step's sentence says what it did with the values chosen; methods.py joins them
into the methods paragraph (`@sentence("dff", clause=True)` is a clause of the sentence that says how the dF/F was made, the others are sentences of their own). Adding an option is editing its step's declaration and the code that obeys it; the app's
Pipeline dialog, the saved state, the provenance record, the command line's `--pipeline` and the methods text all read this registry.

Every option's default is what the pipeline did before the option existed, bit for bit (tests/test_pipeline_options.py compares against a frozen copy of the old
functions), so a recording saved before the options replays unchanged.

    options = PipelineOptions({"denoise.cutoff_hz": 8})     # the rest are defaults; ValueError says what is wrong with a value
    options["denoise.cutoff_hz"]                            # 8.0
    options.to_dict()                                       # every option, {"dff.f0_method": "double_exponential", ...}: what is saved and recorded
    options.changed()                                       # only the ones that are not the default
    options_of(cache)                                       # the options a loaded recording was processed with
"""

import math
from dataclasses import dataclass

_KINDS = ("choice", "bool", "float", "int", "seconds_or_none")
_TRUE, _FALSE = ("true", "yes", "on", "1"), ("false", "no", "off", "0")


@dataclass(frozen=True)
class Option:
    """One option of a step. `kind` says what a value is: "choice" (one of `choices`), "bool", "float" or "int" (between `minimum` and `maximum`, inclusive,
    where given) or "seconds_or_none" (a time in seconds, or None for "not set"). `label` and `doc` are what a person reads in the app's dialog."""
    step: str
    key: str
    kind: str
    default: object
    label: str
    doc: str = ""
    choices: tuple = ()
    minimum: float = None
    maximum: float = None
    unit: str = ""

    def __post_init__(self):
        if self.kind not in _KINDS:
            raise ValueError(f"option {self.name}: kind must be one of {', '.join(_KINDS)}, not {self.kind!r}")
        if self.kind == "choice" and self.default not in self.choices:
            raise ValueError(f"option {self.name}: the default {self.default!r} is not one of its choices {self.choices}")

    @property
    def name(self):
        return f"{self.step}.{self.key}"

    def check(self, value):
        """`value` as this option holds it (a float for a float, ...), or ValueError that says what is wrong."""
        name = self.name
        if self.kind == "bool":
            if not isinstance(value, bool):
                raise ValueError(f"{name} is true or false, not {value!r}")
            return value
        if self.kind == "choice":
            if value not in self.choices:
                raise ValueError(f"{name} must be one of {', '.join(map(str, self.choices))}, not {value!r}")
            return value
        if value is None:
            if self.kind == "seconds_or_none":
                return None
            raise ValueError(f"{name} needs a number")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} is a number{' of ' + self.unit if self.unit else ''}, not {value!r}")
        if self.kind == "int":
            if value != int(value):
                raise ValueError(f"{name} is a whole number, not {value!r}")
            value = int(value)
        else:
            value = float(value)
        if self.minimum is not None and value < self.minimum:
            raise ValueError(f"{name} must be at least {self.minimum:g}, not {value:g}")
        if self.maximum is not None and value > self.maximum:
            raise ValueError(f"{name} must be at most {self.maximum:g}, not {value:g}")
        return value

    def parse(self, text):
        """`text` (what a command-line flag or a text box holds) as a value of this option; ValueError if it is not one."""
        text = str(text).strip()
        if self.kind == "bool":
            low = text.lower()
            if low in _TRUE:
                return self.check(True)
            if low in _FALSE:
                return self.check(False)
            raise ValueError(f"{self.name} is true or false, not {text!r}")
        if self.kind == "choice":
            return self.check(text)
        if self.kind == "seconds_or_none" and text.lower() in ("", "none", "off"):
            return None
        try:
            number = float(text)
        except ValueError:
            raise ValueError(f"{self.name} is a number{' of ' + self.unit if self.unit else ''}, not {text!r}") from None
        return self.check(number)


@dataclass
class Step:
    """One step of the pipeline: its options, what makes a set of them inconsistent (`check(values)` raises ValueError), and the sentence it adds to the methods
    paragraph (`sentence(options, state)`, set with @sentence)."""
    name: str
    title: str
    order: int
    options: tuple
    check: object = None
    sentence: object = None
    clause: bool = False             # the sentence is a clause of the one that says how the dF/F was made (methods.py), not a sentence of its own
    sources: tuple = None            # the kinds of recording the step applies to ("TDT"); None: all of them


STEPS = {}          # step name -> Step
OPTIONS = {}        # "step.key" -> Option


def register_step(name, title, order, options=(), check=None, sources=None):
    """Declares a step of the pipeline and its options (see the module docstring). `sources` are the kinds of recording the step applies to (a recording's
    cache["source"]: "TDT"; the motion correction and the dF/F only exist for TDT), None for all of them. Declaring it again replaces it (a module can be imported
    twice in a test run)."""
    options = tuple(options)
    for option in options:
        if option.step != name:
            raise ValueError(f"option {option.name} is declared in step {name!r}")
    previous = STEPS.get(name)
    for old in previous.options if previous else ():
        OPTIONS.pop(old.name, None)
    STEPS[name] = Step(name, title, order, options, check, previous.sentence if previous else None, previous.clause if previous else False,
                       tuple(sources) if sources else None)
    OPTIONS.update({o.name: o for o in options})
    return STEPS[name]


def sentence(step_name, clause=False):
    """A decorator: the function it wraps, `sentence(options, state) -> str or None`, is what `step_name` adds to the methods paragraph (None or "": it adds nothing
    for these options or this recording). `options` is a PipelineOptions; `state` is a dict of what is known of the recording and the software (methods.facts_of).
    With `clause`, what it returns is a clause of the sentence that says how the dF/F was made ("dF/F was computed by ..., then divided by ..."); without, a
    sentence of its own."""
    def register(function):
        STEPS[step_name].sentence = function
        STEPS[step_name].clause = clause
        return function
    return register


def ordinal(n):
    """1 -> "1st", 2 -> "2nd", 3 -> "3rd", 4 -> "4th", 11 -> "11th"."""
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def ensure_loaded():
    """Imports the modules that declare steps (a step is declared where its code is, so the registry is complete only once they are imported)."""
    from . import recording                    # noqa: F401
    from .sources import processing_TDT       # noqa: F401
    from .analysis import shared, zscore_peth  # noqa: F401


def steps():
    """Every step, in the order of the pipeline."""
    ensure_loaded()
    return sorted(STEPS.values(), key=lambda s: s.order)


def all_options():
    """Every option, in the order of the pipeline's steps (and within one, as declared)."""
    return [o for step in steps() for o in step.options]


def defaults():
    """{name: default} of every option."""
    return {o.name: o.default for o in all_options()}


class PipelineOptions:
    """The value of every option: those given, checked, and the defaults for the rest. Immutable in use; `with_values` makes another."""

    _defaults = None

    def __init__(self, values=None):
        ensure_loaded()
        values = dict(values or {})
        unknown = sorted(set(values) - set(OPTIONS))
        if unknown:
            raise ValueError(f"unknown pipeline option{'s' if len(unknown) > 1 else ''} {', '.join(unknown)}; the options are: {', '.join(OPTIONS)}")
        self._values = {name: (option.check(values[name]) if name in values else option.default) for name, option in OPTIONS.items()}
        for step in STEPS.values():
            if step.check is not None:
                step.check(self._values)

    @classmethod
    def from_dict(cls, values):
        """The options of a dict, a PipelineOptions (as it is), or None (the defaults, made once: an analysis asks for them for every trial)."""
        if isinstance(values, cls):
            return values
        if values is None:
            if cls._defaults is None:
                cls._defaults = cls()
            return cls._defaults
        return cls(values)

    def __getitem__(self, name):
        try:
            return self._values[name]
        except KeyError:
            raise KeyError(f"no pipeline option {name!r}") from None

    def __eq__(self, other):
        return isinstance(other, PipelineOptions) and self._values == other._values

    def __repr__(self):
        changed = self.changed()
        return f"PipelineOptions({changed})" if changed else "PipelineOptions(defaults)"

    def to_dict(self):
        """Every option's value, {name: value}: what is saved with a recording and written in a provenance record."""
        return dict(self._values)

    def changed(self):
        """Only the options whose value is not their default."""
        return {name: v for name, v in self._values.items() if v != OPTIONS[name].default}

    def is_default(self):
        return not self.changed()

    def with_values(self, values):
        """These options with `values` ({name: value}) put on top."""
        return PipelineOptions({**self._values, **dict(values or {})})

    def step_values(self, step):
        """{key: value} of one step's options, as the step's code reads them."""
        return {o.key: self._values[o.name] for o in STEPS[step].options}


def describe():
    """The registry as plain data, for a program that shows or asks for the options (the app's Pipeline dialog, `python -m PyATCLI pipeline`):
    {"steps": [{"name", "title", "sources", "options": [{"name", "key", "kind", "default", "label", "doc", "choices", "minimum", "maximum", "unit"}]}],
    "defaults": {name: default}}. A step's "sources" lists the kinds of recording it applies to (["TDT"]); None: all of them."""
    return {"steps": [{"name": step.name, "title": step.title, "sources": list(step.sources) if step.sources else None,
                       "options": [{"name": o.name, "key": o.key, "kind": o.kind, "default": o.default, "label": o.label, "doc": o.doc, "choices": list(o.choices),
                                    "minimum": o.minimum, "maximum": o.maximum, "unit": o.unit} for o in step.options]} for step in steps()],
            "defaults": defaults()}


def parse_assignments(items):
    """{name: value} from command-line style "name=value" strings (None or []: {}); the value is parsed by the option, so "denoise.enabled=no" is False.
    ValueError that says what is wrong for a string that is not name=value, a name that is not an option, or a value the option does not accept."""
    ensure_loaded()
    out = {}
    for item in items or []:
        name, sep, text = str(item).partition("=")
        name = name.strip()
        if not sep:
            raise ValueError(f"{item!r} is not name=value (for example denoise.cutoff_hz=8)")
        if name not in OPTIONS:
            raise ValueError(f"unknown pipeline option {name!r}; the options are: {', '.join(OPTIONS)}")
        out[name] = OPTIONS[name].parse(text)
    return out


def options_of(cache):
    """The options a loaded recording was processed with (its cache's "pipeline"); the defaults for a cache that has none (one made before the options existed)."""
    return PipelineOptions.from_dict((cache or {}).get("pipeline"))
