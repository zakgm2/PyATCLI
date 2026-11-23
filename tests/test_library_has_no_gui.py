"""The library computes and the app shows it: no module of PyATCLI imports a plotting or window toolkit — not at the top, and not inside a function that a run
might never reach (the subprocess tests in test_cli_conventions.py and test_results.py only see what a run touched). A figure is the app's: PyAT's figures package
draws from the numbers the library hands over."""

import ast
import os
import unittest

PACKAGE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "PyATCLI")
TOOLKITS = {"matplotlib", "PyQt5", "PyQt6", "PySide2", "PySide6", "tkinter", "pyqtgraph", "seaborn", "plotly", "bokeh"}


def imported_toolkits(tree):
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module]
        else:
            continue
        found += [(node.lineno, n) for n in names if n.split(".")[0] in TOOLKITS]
    return found


class TestNoGuiInTheLibrary(unittest.TestCase):
    def test_no_module_imports_a_plotting_or_window_toolkit(self):
        offenders = []
        for folder, dirs, files in os.walk(PACKAGE):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for name in files:
                if name.endswith(".py"):
                    path = os.path.join(folder, name)
                    with open(path, encoding="utf-8") as f:
                        for line, module in imported_toolkits(ast.parse(f.read())):
                            offenders.append(f"{os.path.relpath(path, PACKAGE)}:{line} imports {module}")
        self.assertEqual(offenders, [], "the library does not draw: move the drawing to the app (PyAT's figures package) and keep the numbers here")

    def test_the_check_sees_a_lazy_import_inside_a_function(self):
        tree = ast.parse("def f():\n    import matplotlib.pyplot as plt\n    from PyQt6 import QtCore\n")
        self.assertEqual([m for _, m in imported_toolkits(tree)], ["matplotlib.pyplot", "PyQt6"])


if __name__ == "__main__":
    unittest.main()
