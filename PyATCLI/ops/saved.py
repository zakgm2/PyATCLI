"""ops/saved.py — what the app's Save Changes saves next to a recording, read and written."""

from .. import saved_state
from ..recording import HIGHLIGHTS_FILE, TEXT_ANNOTATIONS_FILE
from . import OpError, operation
from .common import load_options, recording_name, require_exists


def _edit_options(load):
    """The load options an edit needs (it opens the recording as saved; a saved edit never skips the saved state)."""
    options = load_options(load)
    options.pop("use_saved", None)
    return options


@operation("saved.show", "list what is saved for a recording")
def show(ctx, *, recording):
    path = require_exists(recording)
    summary = saved_state.saved_state_summary(path)
    return {"recording": path, "name": recording_name(path), **summary}


@operation("saved.marker", "Add Marker + Save Changes: put a marker at a time (seconds, on the time axis as the recording now stands)")
def marker(ctx, *, recording, time, label=None, color=None, fontsize=None, load=None):
    m = saved_state.add_marker(require_exists(recording), time, label, color, fontsize, **_edit_options(load))
    ctx.written(saved_state.marker_sidecar_path(recording))
    return {"marker": m}


@operation("saved.unmark", "delete the saved markers near a time")
def unmark(ctx, *, recording, time, tolerance=0.05, label=None, load=None):
    removed = saved_state.remove_markers(require_exists(recording), time, tolerance, label, **_edit_options(load))
    if removed:
        ctx.written(saved_state.marker_sidecar_path(recording))
    return {"removed": removed}


@operation("saved.splice", "Splice + Save Changes: stack a splice on the saved ones (it is replayed whenever the recording opens)")
def splice(ctx, *, recording, mode, start, end, load=None):
    splices = saved_state.add_splice(require_exists(recording), mode, start, end, **_edit_options(load))
    ctx.written(saved_state.splice_sidecar_path(recording))
    return {"splices": splices}


@operation("saved.unsplice", "take one saved splice out of the stack")
def unsplice(ctx, *, recording, number):
    left = saved_state.remove_splice(require_exists(recording), number)
    return {"splices": left}


@operation("saved.rename", "give signals names of your own, kept with the recording")
def rename(ctx, *, recording, names, load=None):
    saved = saved_state.rename_signals(require_exists(recording), names, **_edit_options(load))
    ctx.written(saved_state.signal_labels_path(recording))
    return {"names": saved}


@operation("saved.highlight", "Highlighter + Save Changes: shade a stretch of the recording")
def highlight(ctx, *, recording, start, end, color="Yellow"):
    h = saved_state.add_highlight(require_exists(recording), start, end, color)
    return {"highlight": h}


@operation("saved.text", "Text tool + Save Changes: put a text box on the plot")
def text(ctx, *, recording, x, y, text, fontsize=saved_state.DEFAULT_TEXT_FONT_SIZE):
    t = saved_state.add_text(require_exists(recording), x, y, text, fontsize)
    return {"text": t}


@operation("saved.clear", "Undo All Changes, saved: delete every file in the recording's JSON saves folder (yes=True to go ahead)", result=["removed"])
def clear(ctx, *, recording, yes=False):
    path = require_exists(recording)
    if not yes:
        summary = saved_state.saved_state_summary(path)
        n = sum(bool(summary[k]) for k in ("markers", "splices", "signal_labels", "highlights", "text_annotations"))
        raise OpError(f"this deletes every file in {summary['folder']} ({n} kinds of saved edit)", "confirmation_required",
                      hint="say --yes to do it")
    return {"removed": saved_state.clear_saved_state(path)}


@operation("saved.write", "Save Changes: write whole lists — markers, splices, signal names, highlights, text boxes — replacing what was saved", result=["files"])
def write(ctx, *, recording, markers=None, splices=None, signal_labels=None, highlights=None, text_annotations=None):
    path = require_exists(recording)
    files = []
    if markers is not None:
        files.append(saved_state.write_saved_markers(path, markers))
    if splices is not None:
        files.append(saved_state.write_saved_splices(path, splices))                  # None: no splices, the file is removed
    if signal_labels is not None:
        files.append(saved_state.write_saved_signal_labels(path, signal_labels))
    if highlights is not None:
        files.append(saved_state.write_saved_annotations(path, HIGHLIGHTS_FILE, highlights))
    if text_annotations is not None:
        files.append(saved_state.write_saved_annotations(path, TEXT_ANNOTATIONS_FILE, text_annotations))
    files = [f for f in files if f]
    for f in files:
        ctx.written(f)
    return {"files": files}
