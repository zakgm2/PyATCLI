# Contributing to PyATCLI

Thank you for looking. PyATCLI is a small project with one maintainer, so here is what to expect and how to make a change easy to accept.

## Reporting a problem or asking a question
- Open an issue at https://github.com/zakgm2/PyATCLI/issues. For a problem, say which version you have (`python -m PyATCLI info`), what you ran, what you expected, and what happened. A short script that reproduces a wrong number is the most useful thing you can send.
- Please do not post data you are not allowed to share.
- Expect a first reply within about a week. This is research software maintained alongside a degree; replies can be slower in exam periods.

## Changing the code
1. Fork, branch from `main`, and keep the change to one thing.
2. Install what the tests read (`tdt` and `openpyxl` are extras; `sentence-transformers` is left out, the tests do not touch it and it pulls in `torch`):
   ```bash
   pip install -e ".[tdt,excel]"
   ```
   Then run the tests before you open a pull request:
   ```bash
   python -m unittest discover -s tests -t .
   ```
   They take about a minute. CI runs them on Windows and Linux, Python 3.10 and 3.13 (without matplotlib installed: the library draws nothing), and also runs the PyAT app's tests against your branch.
3. A fix should come with a test that fails without it; a new feature should come with tests of what it does and a section in `API_REFERENCE.md`.
4. The generated files must stay in step: after changing a command or an operation run
   ```bash
   python -m PyATCLI docs --format markdown --out CLI_REFERENCE.md
   python -m PyATCLI docs --format man --out man/pyatcli.1
   python -m PyATCLI docs --format contract --out ENGINE_CONTRACT.json
   ```
   The tests fail on drift. `tests/contracts/protocol_1.json` is the engine protocol as it was released: never edit it; a change that breaks it needs a new protocol version.
5. Everything is computed in float64 (`PyATCLI.numeric.as_float64`); keep it that way, and do not add drawing code (figures are drawn by the PyAT app).
6. Match the surrounding code: names, comment density, how errors are reported.
7. Add a line under `## Unreleased` in `CHANGELOG.md`.
8. **Adding a pipeline option.** A choice the processing makes goes in the registry (`PyATCLI/pipeline.py`), declared where the step's code is:
   - add a `pipeline.Option(step, key, kind, default, label, doc, ...)` to the step's `register_step(...)` (a new step is a new `register_step` call with an `order`), and read the value where the code uses it
     (`options["denoise.cutoff_hz"]`). The default must reproduce what the code did before, bit for bit: `tests/legacy_pipeline.py` is a frozen copy of the old functions, and
     `tests/test_pipeline_options.py` compares against it — add the new option's default to that comparison;
   - give the step a sentence for the methods paragraph with `@pipeline.sentence(step)` (`clause=True` for a clause of the sentence about the dF/F), saying the value chosen;
   - add the option to `CHANGES` in `tests/test_methods_text.py` with what the paragraph must say (it fails until you do), and a test of what the option does to the numbers.
   Nothing else is edited: the saved state (`JSON saves/pipeline.json`), the provenance record, the command line's `--pipeline`, the engine's `pipeline.options` and the app's Pipeline dialog all read the
   registry. Regenerate the generated files (4) if you added an operation.

## Licence
By contributing you agree that your contribution is released under the project's licence, AGPL-3.0-only (see `LICENSE`).

## Citing
See `CITATION.cff`, or the "Cite this repository" button on GitHub.
