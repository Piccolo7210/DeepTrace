"""
Entry point for the desktop app.

Launch either as:
    (from the project root)  venv312\\Scripts\\python -m app
    (from anywhere)          venv312\\Scripts\\python <project_root>\\app\\__main__.py

Both work regardless of the current working directory: src/config.py resolves its
paths (THRESHOLD_PATH, RESULTS_DIR, ...) from its own file location rather than the
CWD, and this file puts app/ on sys.path itself rather than relying on `-m`'s
package-import machinery.
"""

import sys
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

import matplotlib

matplotlib.use("Agg")

from ui.app_window import AppWindow


def main():
    app = AppWindow()
    app.mainloop()


if __name__ == "__main__":
    main()
