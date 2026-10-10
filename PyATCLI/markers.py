"""
markers.py
----------
How an event marker is named and how markers are grouped by name — shared by every analysis that
works "per event" (Event PETH, the peak finder, group analysis) so they all agree on what a marker is
called. A marker is a dict with at least a 'time' (seconds) and a 'label'; the ones TDT detected also
carry 'store' (the epoc store's id, e.g. 'PP1_') and, for onset/offset pairs, 'phase' ('high'/'low') —
see processing_TDT.get_event_markers.

The raw store id stays the internal identifier everywhere; only how it is *shown* can be renamed
(`store_labels`, {store id: shown name}). Group analysis saves the renames it was run with, because a
marker name only means something together with them.
"""

PHASE_SUFFIX = {'high': '¹', 'low': '⁰'}  # superscript 1 / 0


def store_display_name(store_id, store_labels=None):
    """The name shown for a store id — the renamed name if one has been set in `store_labels`,
    otherwise the raw store id unchanged."""
    if store_id is None:
        return store_id
    return (store_labels or {}).get(store_id, store_id)


def marker_display_label(marker, store_labels=None):
    """Full text to show for a marker: its store's renamed display name, plus a superscript ¹/⁰ if
    it's a high/low phase marker.

    Only markers with a 'phase' key get the store-name substitution — those are exactly the
    auto-generated onset/offset markers whose label IS the store name by convention (see
    processing_TDT.get_event_markers). Everything else (Note markers, whose label is the actual note
    text like 'Clap' even though they share store == 'Note'; manually-placed markers) keeps its own
    label untouched — renaming a store must not clobber a marker whose label was never derived from
    that store name."""
    if marker.get('phase') is not None:
        label = store_display_name(marker['store'], store_labels)
    else:
        label = marker['label']
    return f"{label}{PHASE_SUFFIX.get(marker.get('phase'), '')}"


def group_markers_by_name(cache, store_labels=None):
    """{display_name: [sorted event times]} across every event TDT actually detected in this
    recording (cache['detected_markers']), plus anything manually placed or already added to the plot
    (cache['markers']) — deduped by (name, time) so a marker that's both detected and already plotted
    doesn't get double-counted. This intentionally does NOT require an event to already be on the
    plot: the whole point is finding out which event types are worth plotting in the first place.
    High/low phase markers count as separate event names (their display label already differs by the
    ¹/⁰ suffix).

    Parameters
    ----------
    cache : dict
        A recording as PyAT holds it (see recording.py): only 'detected_markers' (optional) and
        'markers' are read.
    store_labels : dict, optional
        {store id: shown name} renames in effect.
    """
    groups = {}
    seen = set()
    for m in cache.get('detected_markers', []) + cache['markers']:
        name = marker_display_label(m, store_labels)
        key = (name, m['time'])
        if key in seen:
            continue
        seen.add(key)
        groups.setdefault(name, []).append(m['time'])
    for name in groups:
        groups[name].sort()
    return groups
