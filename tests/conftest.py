import sys
from pathlib import Path

import matplotlib


def pytest_configure():
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    matplotlib.use("Agg")

