"""
PyATCLI
-------
The analysis library, engine and command line behind PyAT (Python Analysis Tool): loading TDT, Oxysoft, generic-table and PT2 data, the analyses
(Z-score, AUC, FFT, curve fit, Event PETH, peaks, group statistics), saved state, provenance records, and `python -m PyATCLI`. It computes and
writes numbers and draws nothing; PyAT draws them. (Called PhysicsLibrary, on PyPI ZaksPhysicsLibrary, before it became PyATCLI.)
"""

from ._version import resolve_version as _resolve_version

__version__ = _resolve_version()   # a source checkout reports its own pyproject version — see _version.py

from .sources.tables import load_any_file

from .progress import Plan, track

from .provenance import (
    SCHEMA_VERSION as PROVENANCE_SCHEMA_VERSION,
    SIDECAR_SUFFIX as PROVENANCE_SUFFIX,
    software_versions,
    library_version,
    check_environment,
    pipeline_parameters,
    file_fingerprint,
    markers_summary,
    recording_state,
    build_record,
    output_entry,
    sidecar_path_for,
    write_record,
    read_record,
)

from .sources.dataset import detect_format, detect_format_file, DataFormat, Dataset

from .sources.load import load_dataset, load_dataset_file

from .sources.pt2 import load_pt2

from .sources.processing_oxysoft import process_oxysoft_folder, process_oxysoft_file

from .sources.processing_TDT import (
    process_tdt_folder,
    compute_dff,
    validate_tdt_folder,
    get_tdt_struct,
    scan_tdt_markers,
    get_plot_data,
    correct_bleaching,
    denoise_signal,
    get_event_markers,
    debounce_events,
    REGRESSION_METHODS,
)

from .analysis import (
    estimate_sample_rate,
    mean_channels,
    compute_group_stats,
    compute_marker_intervals,
    get_zscore_slice,
    smooth_signal,
    bin_for_heatmap,
    compute_fft_slice,
    find_fft_peaks,
    compute_slope_segment,
    fit_model_to_segment,
    compute_event_zscore_peth,
    find_significant_peaks,
    find_peak_near_events,
    compute_auc_from_trace,
    compute_auc_window,
    compute_auc_matrix,
    compute_peri_event_from_trace,
    compute_peri_event_matrix,
    GroupSpec,
    marker_index,
    common_markers,
    design_summary,
    extract_group_trials,
    measures_for_trial,
    decay_time,
    fit_group_models,
    GroupResults,
    write_group_results,
    group_report_text,
    default_subject_names,
    scan_recording,
    scan_recordings,
    GROUP_SETTINGS_FILENAME,
    load_group_recording,
    run_group_analysis,
    write_group_settings,
    read_group_settings,
)

from .splice import (
    splice_keep_inside,
    splice_cut_out,
    replay_splices,
    MODE_CUT_OUT,
    MODE_KEEP_INSIDE,
)

from .markers import (
    PHASE_SUFFIX,
    store_display_name,
    marker_display_label,
    group_markers_by_name,
)

from .recording import (
    SAVES_FOLDER,
    saved_state_dir,
    marker_sidecar_path,
    legacy_marker_sidecar_path,
    annotation_sidecar_path,
    splice_sidecar_path,
    read_saved_splices,
    read_saved_markers,
    read_saved_annotations,
    signal_labels_path,
    read_saved_signal_labels,
    apply_signal_labels,
    current_signal_labels,
    build_signal_options,
    make_tdt_cache,
    make_oxysoft_cache,
    make_generic_cache,
    normalized_signal,
    apply_splice,
    apply_splices,
    spliced_store_name,
    load_tdt_recording,
    load_recording,
    find_tdt_recordings,
)

from .models import (
    double_exponential_model,
    visibility_model,
    linear_model,
    single_exponential_model,
    exponential_rise_model,
    gaussian_model,
    sinusoidal_model,
    CURVE_FIT_MODELS,
)

from .fieldstudy.study import (
    run_field_study_pipeline,
    load_field_study_folder,
    peek_fields,
    flag_low_quality,
    embed_text_fields,
    compute_delta_vector,
    compute_paired_similarity,
    permutation_test_similarity,
    wordcount_confound_check,
)

from .fieldstudy.validation import (
    run_validation_pipeline,
    build_validation_summary,
    cohens_d,
    benjamini_hochberg,
    wordcount_controlled_regression,
    bootstrap_mean_ci,
    leave_one_out_sensitivity,
)

# Scripted use: run any analysis from a recording and a parameter dict, and run one again from its provenance record.
# `reproduce` stays the module (PyATCLI.reproduce); its function is reproduce_record.
from .analyses import (
    run_analysis,
    kinds,
    describe_kinds,
    normalize_parameters,
    AnalysisResult,
    ParameterError,
    AnalysisError,
)

from .reproduce import (
    reproduce as reproduce_record,
    compare_table,
    Reproduction,
    ReproduceError,
)

# Where results go once there is an Output folder (the same layout PyAT's exports use)
from . import outputs, saved_state

# What an Export View holds, as numbers (PyAT draws the figure; nothing in this library draws)
from .analyses import VIEW_KIND, run_view
from .results import NoResults, OpenedResults, open_results
