"""
viewplan.py
-----------
What the main plot of a recording shows, as data: which lines it has (under what name, with what numbers), the range it is looking at, the markers
in it, and the title and axis labels it would carry. The numbers of an Export View are this — `analyses.run_view` replays them without a window or a
plotting library, and a viewer (PyAT's main plot, both its engines, and Export View) draws it.

    plan = viewplan.plan_view(cache, signals=["normalized", "main_driver"], xlim=(40, 120))
    plan.data_table()        every sample in view, the time and a column per line
    plan.markers_table()     the markers in view
    plan.parameters()        what it takes to make this plan again (a provenance record's parameters)

How a line is drawn — its colour, width, marker, the figure it ends up in — is the viewer's business: a `Trace` carries the `role` it plays
("signal", "o2hb_mean", "column", ...) and the `signal` key and `color` the recording itself gives it, and the viewer styles it from those.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .analysis.shared import mean_channels
from .markers import marker_display_label

FULL_VIEW_PADDING_FRAC = 0.02         # 2% of the data span on each side of a "show everything" view

# What a line is, for a viewer that styles by it.
ROLE_SIGNAL = "signal"                # a TDT signal (`signal` is its key in the recording's `signals`)
ROLE_CORRECTED = "corrected"          # the corrected trace of a source with no signals map
ROLE_O2HB_CHANNEL, ROLE_HHB_CHANNEL = "o2hb_channel", "hhb_channel"          # one NIRS channel
ROLE_O2HB_MEAN, ROLE_HHB_MEAN, ROLE_THB_MEAN = "o2hb_mean", "hhb_mean", "thb_mean"    # the mean of the NIRS channels
ROLE_COLUMN = "column"                # one column of a generic table, in the order the table has them


def padded_xlim(x):
    """(lo, hi) for "fit the full trace with a little breathing room": an exact-bounds range leaves the trace flush against the edges of the
    plot. Used for every "show everything" moment."""
    lo, hi = float(x[0]), float(x[-1])
    pad = (hi - lo) * FULL_VIEW_PADDING_FRAC
    if pad == 0:
        pad = 1.0                  # degenerate single-point/flat data — avoid a zero-width range
    return lo - pad, hi + pad


@dataclass
class Trace:
    """One line of the main plot. `x`/`y` are what is shown; `y_full` the same signal over the recording's whole time axis (NaN where there
    is none), which is what the data CSV is made of. `label` is the legend name ("_nolegend_" for a channel that follows its group's
    entry) and `key` the name a viewer files per-line choices under (a group's unnamed lines follow the group's)."""
    label: str
    x: np.ndarray
    y: np.ndarray
    csv_name: str
    y_full: np.ndarray = None
    key: str = None
    role: str = ROLE_SIGNAL
    signal: str = None                 # a TDT signal's key in cache["signals"]
    color: str = None                  # the colour the recording itself gives the signal, if it gives one


def view_traces(cache, *, signals=None, active=None):
    """(traces, title, x label, y label) for what the main plot shows for a recording, before any edit of its title, labels or legend.

    signals       a TDT recording's signals ticked in the Plot menu (keys of cache["signals"]: "normalized", "main_driver",
                  "isosbestic", ...); more than one are overlaid, in the recording's order. Default: "normalized".
    active        the signal shown when only one is — default: the one ticked (or the first the recording has)
    """
    x = cache['x']
    source = cache.get('source')
    traces = []

    def add(label, y, role, csv_name=None, key=None, **kw):
        traces.append(Trace(label, kw.pop("x", x), y, csv_name or label, y_full=kw.pop("y_full", y), key=key if key is not None else label,
                            role=role, **kw))

    if source == 'Oxysoft':
        o2hb, hhb = cache['o2hb'], cache['hhb']
        for i in range(o2hb.shape[0]):
            add('O2Hb channels' if i == 0 else '_nolegend_', o2hb[i], ROLE_O2HB_CHANNEL, f"O2Hb ch{i + 1}", key='O2Hb channels')
            add('HHb channels' if i == 0 else '_nolegend_', hhb[i], ROLE_HHB_CHANNEL, f"HHb ch{i + 1}", key='HHb channels')
        ff = cache.get('fit_factor_mean')
        ff_tag = f"  [FF: {ff:.1f}%]" if ff is not None else ""
        add(f'Mean O2Hb{ff_tag}', mean_channels(o2hb), ROLE_O2HB_MEAN)
        add(f'Mean HHb{ff_tag}', mean_channels(hhb), ROLE_HHB_MEAN)
        if 'thb' in cache:
            add(f'Mean tHb{ff_tag}', mean_channels(cache['thb']), ROLE_THB_MEAN)
        return traces, f"NIRS — {cache['store']}", "Time (s)", "Delta Concentration (uM)"

    if source == 'Generic':
        for col_name, y in cache['y_columns'].items():
            mask = ~np.isnan(y)
            add(col_name, y[mask], ROLE_COLUMN, x=x[mask], y_full=y)
        return traces, cache['store'], cache.get('x_label', 'X'), "Value"

    available = cache.get('signals') or {}
    ticked = list(signals) if signals is not None else ["normalized"]
    if source == 'TDT' and len(ticked) > 1:
        checked = [(k, sig) for k, sig in available.items() if k in ticked]
        if not checked and available:                     # nothing ticked that the recording has: the one a single view would show
            key = active if active in available else next(iter(available))
            checked = [(key, available[key])]
        for key, sig in checked:
            add(sig['label'], sig['y'], ROLE_SIGNAL, signal=key, color=sig.get('color'))
        return traces, f"Overlay ({len(checked)}) — {cache['store']}", "Time (s)", "Amplitude"

    if not available:                                     # a source with no signals map: its corrected trace, else its raw one
        add("dF/F (corrected)", (cache['corr'] if 'corr' in cache else cache['raw']), ROLE_CORRECTED)
        label = "dF/F (corrected)"
    else:
        want = active if active is not None else (ticked[0] if ticked else "normalized")
        key = want if want in available else next(iter(available))
        label = available[key]['label']
        add(label, available[key]['y'], ROLE_SIGNAL, signal=key, color=available[key].get('color'))
    return traces, f"{label} — {cache['store']}", "Time (s)", "Amplitude"


@dataclass
class ViewPlan:
    """A view of a recording: the lines (`traces`), the range, the markers in it. `attributes` is whatever the viewer says it was drawn with
    (a title, labels, colours...): carried into `parameters()` as it is, never interpreted here."""
    traces: list
    xlim: tuple
    ylim: tuple
    x_name: str
    x: np.ndarray
    markers: list = field(default_factory=list)       # [(time, label, colour)] in view
    signals: list = None                              # a TDT recording's signals shown, in the recording's order
    show_grid: bool = True
    title: str = ""
    x_label: str = ""
    y_label: str = ""
    in_view: list = field(default_factory=list)       # the marker dicts in view
    attributes: dict = field(default_factory=dict)

    def parameters(self):
        """What it takes to make this view again (analyses.run_view): the range, the signals, the grid and the attributes. JSON-ready."""
        return {"x_range": list(self.xlim), "y_range": list(self.ylim) if self.ylim is not None else None, "signals": self.signals,
                "show_grid": bool(self.show_grid), "attributes": self.attributes, "traces": [t.csv_name for t in self.traces]}

    def data_table(self):
        """One row per sample in view: the time (or x) and a column per trace, at the recording's full resolution."""
        lo, hi = self.xlim
        mask = (self.x >= lo) & (self.x <= hi)
        columns = {self.x_name: self.x[mask]}
        for t in self.traces:
            name, n = t.csv_name, 2
            while name in columns:
                name, n = f"{t.csv_name} ({n})", n + 1
            columns[name] = np.asarray(t.y_full)[mask]
        return pd.DataFrame(columns)

    def markers_table(self):
        """The markers in view: time, shown name, colour (no rows if there are none)."""
        return pd.DataFrame(self.markers, columns=[self.x_name, "label", "color"])

    def write_data(self, base):
        """<base>_data.csv (every sample in view) and, if any marker is in view, <base>_markers.csv. Returns the paths."""
        paths = [f"{base}_data.csv"]
        self.data_table().to_csv(paths[0], index=False, encoding="utf-8")
        if self.markers:
            paths.append(f"{base}_markers.csv")
            self.markers_table().to_csv(paths[1], index=False, encoding="utf-8")
        return paths


def plan_view(cache, *, signals=None, active=None, xlim=None, ylim=None, show_grid=True, store_labels=None, attributes=None):
    """What the main plot shows of `cache` (a recording, as load_recording returns it): the traces, the range, the markers in it, the title and
    labels. `signals`, `active`: see view_traces. `xlim`, `ylim`: the range, (low, high); default all of the recording with a little room,
    and the y range follows the data. `store_labels`: {store id: shown name}, the renames in force for marker names. `attributes`: what the
    viewer drew it with, kept for `parameters()`."""
    traces, title, x_label, y_label = view_traces(cache, signals=signals, active=active)
    lo, hi = (float(min(xlim)), float(max(xlim))) if xlim is not None else padded_xlim(cache['x'])
    ylim = (float(min(ylim)), float(max(ylim))) if ylim is not None else None
    in_view = [m for m in cache.get('markers', []) if lo <= m['time'] <= hi]
    x_name = "time_s" if cache.get('source') in ('TDT', 'Oxysoft') else (cache.get('x_label') or 'x')
    markers = [(m['time'], marker_display_label(m, store_labels), m['color']) for m in in_view]
    available = cache.get('signals') or {}
    ticked = list(signals) if signals is not None else ["normalized"]
    shown = [k for k in available if k in ticked] if cache.get('source') == 'TDT' else None
    return ViewPlan(traces, (lo, hi), ylim, x_name, cache['x'], markers, signals=shown, show_grid=bool(show_grid), title=title,
                    x_label=x_label, y_label=y_label, in_view=in_view, attributes=dict(attributes or {}))
