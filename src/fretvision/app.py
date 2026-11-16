"""Application entrypoint for FretVision."""

from __future__ import annotations

import os
import sys

# Prevent conflicting Qt/VC++ DLLs in external Conda installations from breaking PySide6 on Windows.
if sys.platform == "win32" and "PATH" in os.environ:
    os.environ["PATH"] = os.pathsep.join(
        p for p in os.environ["PATH"].split(os.pathsep) if "anaconda" not in p.lower() and "miniconda" not in p.lower()
    )

# Import mediapipe before cv2 via HandTracker to avoid Windows DLL issues.
import fretvision.vision.hand_tracker  # noqa: F401, E402

from PySide6.QtWidgets import QApplication  # noqa: E402

from fretvision.ui.main_window import MainWindow  # noqa: E402


def main() -> int:
    """Launch the FretVision desktop application."""
    app = QApplication(sys.argv)
    app.setApplicationName("FretVision")
    app.setOrganizationName("FretVision")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
