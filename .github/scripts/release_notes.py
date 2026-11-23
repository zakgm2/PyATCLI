"""
Before a release is uploaded: check that the tag that was pushed, pyproject.toml, CITATION.cff and CHANGELOG.md all name the same version, and cut that
version's section out of CHANGELOG.md as the GitHub Release's notes. Run by .github/workflows/publish.yml ahead of the PyPI upload, because PyPI never takes
a version twice (a wrong tag or a missing changelog section found after the upload cannot be undone). tests/test_release_notes.py runs the same checks against
the repository as it is, so a release that would fail here is found by the ordinary tests first.

    python .github/scripts/release_notes.py v2026.11.5 release_notes.md
"""

import re
import sys
import textwrap


def project_version(pyproject_text):
    """The `version = "..."` line of pyproject.toml (read by pattern, not with tomllib, which Python 3.10 does not have)."""
    match = re.search(r'^version\s*=\s*"([^"]+)"\s*$', pyproject_text, re.MULTILINE)
    return match.group(1) if match else None


def citation_version(cff_text):
    match = re.search(r'^version:\s*"?([^"\s]+)"?\s*$', cff_text, re.MULTILINE)
    return match.group(1) if match else None


def changelog_section(changelog_text, version):
    """The text of `## Version <version>...` up to the next `---` line (or the next `## ` heading), dedented; None when there is no such section."""
    lines = changelog_text.splitlines()
    heading = re.compile(rf"^## Version {re.escape(version)}(\s|:|$)")
    start = next((i for i, line in enumerate(lines) if heading.match(line)), None)
    if start is None:
        return None
    body = []
    for line in lines[start + 1:]:
        if line.strip() == "---" or line.startswith("## "):
            break
        body.append(line)
    return textwrap.dedent("\n".join(body)).strip()


def check(tag, pyproject_text, cff_text, changelog_text):
    """(problems, notes): what is wrong, and the release notes (None when something is)."""
    problems = []
    if not re.fullmatch(r"v\d+\.\d+\.\d+", tag):
        return [f"the tag {tag!r} is not of the form vYYYY.M.D"], None
    version = tag[1:]
    if project_version(pyproject_text) != version:
        problems.append(f"the tag says {version} but pyproject.toml says {project_version(pyproject_text)}")
    if citation_version(cff_text) != version:
        problems.append(f"the tag says {version} but CITATION.cff says {citation_version(cff_text)} (Zenodo copies it into the archive)")
    section = changelog_section(changelog_text, version)
    if not section:
        problems.append(f"CHANGELOG.md has no '## Version {version}' section (or it is empty); the GitHub Release's notes come from it")
    if problems:
        return problems, None
    return [], f"{section}\n\n`pip install PyATCLI=={version}` · https://pypi.org/project/PyATCLI/{version}/\n"


def main(tag, out):
    def read(path):
        with open(path, encoding="utf-8") as f:
            return f.read()

    problems, notes = check(tag, read("pyproject.toml"), read("CITATION.cff"), read("CHANGELOG.md"))
    if problems:
        for problem in problems:
            print(f"::error::{problem}")
        return 1
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write(notes)
    print(f"{tag}: pyproject.toml, CITATION.cff and CHANGELOG.md agree; release notes written to {out} ({len(notes.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1], sys.argv[2]))
