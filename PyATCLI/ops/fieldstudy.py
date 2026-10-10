"""
ops/fieldstudy.py — text field studies (a folder of per-participant JSON files whose free-text answers are embedded and compared).

    fieldstudy.fields     the field names in a study folder's files (for choosing which to compare)
    fieldstudy.run        the whole pipeline: load, flag short answers, embed, similarity of each pair of fields, permutation tests
    fieldstudy.validate   the statistical validation of the pairs (permutation p-values with FDR, Cohen's d, bootstrap CI, leave-one-out)

Both pipelines load a language model and embed every answer — a lot to import and hold, and the reason they run in the engine rather than
in a window. The tables come back as the shapes of what they hold, or (tables="files") as CSV files to read.
"""

from ..analyses import ParameterError
from ..fieldstudy.validation import run_validation_pipeline
from ..fieldstudy.study import peek_fields, run_field_study_pipeline
from . import operation
from .common import require_exists, tables_wire


def _pairs(paired_fields):
    """[(field_a, field_b, name), ...] from lists as JSON carries them."""
    try:
        return [(str(a), str(b), str(name)) for a, b, name in (paired_fields or [])]
    except (TypeError, ValueError):
        raise ParameterError("paired_fields is a list of [field_a, field_b, name]") from None


@operation("fieldstudy.fields", "the field names found in the first participant file of a study folder", result=["folder", "fields"])
def fields(ctx, *, folder, file_glob="P-*.json"):
    return {"folder": require_exists(folder), "fields": peek_fields(require_exists(folder), file_glob)}


@operation("fieldstudy.run", "run a text field study: one row per participant, with the similarity of each pair of fields and its tests", result=["folder", "rows", "tables"])
def run(ctx, *, folder, text_fields, paired_fields=None, delta_pair=None, file_glob="P-*.json", min_words=5, tables="shapes"):
    if not text_fields:
        raise ParameterError("text_fields: name the fields to load")
    delta = tuple(str(f) for f in delta_pair) if delta_pair else None
    if delta is not None and len(delta) != 2:
        raise ParameterError("delta_pair is [field_from, field_to]")
    df = run_field_study_pipeline(require_exists(folder), text_fields=[str(f) for f in text_fields], delta_pair=delta,
                                  paired_fields=_pairs(paired_fields) or None, file_glob=file_glob, min_words=int(min_words),
                                  progress=ctx.progress())
    return {"folder": folder, "rows": int(len(df)), "tables": tables_wire(ctx, {"study": df}, tables, prefix="fieldstudy")}


@operation("fieldstudy.validate", "statistical validation of a study's field pairs: one row per pair", result=["folder", "rows", "tables"])
def validate(ctx, *, folder, text_fields, paired_fields, file_glob="P-*.json", tables="shapes"):
    pairs = _pairs(paired_fields)
    if not pairs:
        raise ParameterError("paired_fields: give at least one [field_a, field_b, name] to validate")
    df = run_validation_pipeline(require_exists(folder), text_fields=[str(f) for f in text_fields], paired_fields=pairs, file_glob=file_glob,
                                 progress=ctx.progress())
    return {"folder": folder, "rows": int(len(df)), "tables": tables_wire(ctx, {"validation": df}, tables, prefix="validation")}
