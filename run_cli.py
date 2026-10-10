"""
PyATCLI — command-line entry point
-----------------------------------------
The command line is `python -m PyATCLI` (or `pyatcli`, wherever the library is installed). This file starts it from the repository folder:

    run it with nothing        opens Git Bash in this folder (the same as right-click > "Git Bash Here"), shows the commands there, and leaves the
                               prompt open to type them. Double-click the file, or the shortcut to it, and this is what happens
    run it with arguments      is the command line itself, in the terminal you are in:

        python run_cli.py info
        python run_cli.py run --help
        python run_cli.py run zscore_window "D:/rats/Rat7" -p center_s=312.5

Two switches of its own say which: --terminal always opens the Git Bash window (with or without a command after it), --inline never does (it is what the new
window runs to show the commands, and it is how to see them in a terminal you already have). Without Git for Windows it opens a Command Prompt instead;
anywhere but Windows it is always the command line itself. Every command can print its result as JSON with --json (see CLI_REFERENCE.md).
"""

import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OWN_SWITCHES = ("--terminal", "--inline")


def _say(text):
    """Tells the person something when there may be no console to print it on (the shortcut runs this without one)."""
    if sys.stderr is not None:
        print(text, file=sys.stderr)
    elif os.name == "nt":
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(None, text, "PyATCLI command line", 0x10)
        except Exception:                                                  # noqa: BLE001 — nowhere left to say it
            pass


def _interpreter():
    """The python that has a console (the one this runs under may be pythonw, which prints nothing)."""
    exe = sys.executable
    folder, name = os.path.split(exe)
    if name.lower() == "pythonw.exe" and os.path.exists(os.path.join(folder, "python.exe")):
        return os.path.join(folder, "python.exe")
    return exe


def _git_bash():
    """Where Git for Windows' git-bash.exe is, or None."""
    candidates = []
    git = shutil.which("git")
    if git:                                                                # ...\Git\cmd\git.exe -> ...\Git\git-bash.exe
        candidates.append(os.path.join(os.path.dirname(os.path.dirname(git)), "git-bash.exe"))
    for variable in ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA"):
        base = os.environ.get(variable)
        if base:
            candidates.append(os.path.join(base, "Git", "git-bash.exe"))
            candidates.append(os.path.join(base, "Programs", "Git", "git-bash.exe"))
    return next((c for c in candidates if os.path.exists(c)), None)


def open_terminal(arguments=()):
    """Opens a terminal in this folder that shows the commands (or runs `arguments`) and stays open. Returns 0, or 1 if none could be opened."""
    python = _interpreter().replace("\\", "/")
    script = os.path.join(HERE, "run_cli.py").replace("\\", "/")
    quoted = " ".join("'" + a.replace("'", "'\\''") + "'" for a in arguments)
    startup = (f'"{python}" "{script}" --inline {quoted}; echo; '
               "echo 'Type a command, for example:  python run_cli.py run --help'; exec bash")
    detached = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP    # the window must outlive the console that started it
    bash = _git_bash()
    try:
        if bash:
            subprocess.Popen([bash, f"--cd={HERE}", "-c", startup], creationflags=detached, close_fds=True)
        else:                                                              # no Git for Windows: a Command Prompt will do
            plain = f'"{_interpreter()}" "{os.path.join(HERE, "run_cli.py")}" --inline {subprocess.list2cmdline(list(arguments))}'
            subprocess.Popen(f'cmd.exe /k "{plain}"', cwd=HERE, creationflags=subprocess.CREATE_NEW_CONSOLE)       # (cmd takes the outer quotes off)
    except OSError as e:
        _say(f"A terminal could not be opened: {e}")
        return 1
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    force_terminal = "--terminal" in argv
    force_inline = "--inline" in argv
    argv = [a for a in argv if a not in OWN_SWITCHES]
    in_a_unix_terminal = bool(os.environ.get("MSYSTEM"))                    # Git Bash's own windows: already where it should be
    if os.name == "nt" and not force_inline and (force_terminal or (not argv and not in_a_unix_terminal)):
        return open_terminal(argv)
    try:
        from PyATCLI.cli import main as cli_main
    except ImportError as e:
        _say(f"PyATCLI could not be loaded: {e}\n"
             "Install it for this Python (python -m pip install PyATCLI, or pip install -e <the PyATCLI folder>).")
        return 1
    return cli_main(argv)


if __name__ == "__main__":
    sys.exit(main())
