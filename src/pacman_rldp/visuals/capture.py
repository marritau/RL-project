"""Helpers for capturing Pacman Tk canvas frames and exporting GIFs.

The project used to mix two capture strategies: some algorithms grabbed the Tk
canvas, while food-bitmask VI grabbed the whole desktop.  Whole-screen capture is
fragile on high-DPI displays because it records the complete 2520x1680 desktop
instead of the Pacman canvas.  These helpers always capture the Pacman canvas
bounding box and optionally downscale oversized frames while preserving aspect
ratio.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

from PIL import Image, ImageGrab

from ..third_party.bk import graphicsUtils

Size = Tuple[int, int]
BBox = Tuple[int, int, int, int]


def expected_canvas_size(*, layout_width: int, layout_height: int, zoom: float = 1.0) -> Size:
    """Return the Pacman Tk canvas size for a layout and zoom.

    The Berkeley Pacman renderer uses a 30px grid at zoom=1.0, one grid-cell
    margin on each side, and a 35px info pane.
    """
    grid_size = 30.0 * float(zoom)
    width = int(round((float(layout_width) + 1.0) * grid_size))
    height = int(round((float(layout_height) + 1.0) * grid_size + 35.0))
    return width, height


def canvas_bbox() -> BBox | None:
    """Return absolute screen coordinates of the active Pacman canvas."""
    canvas = graphicsUtils._canvas
    root_window = graphicsUtils._root_window
    if canvas is None or root_window is None:
        return None
    try:
        root_window.update_idletasks()
        root_window.update()
        root_window.deiconify()
        root_window.lift()
    except Exception:
        # Some headless/CI environments cannot manipulate a Tk window.  The
        # caller treats None as "do not record a frame".
        return None

    x0 = int(canvas.winfo_rootx())
    y0 = int(canvas.winfo_rooty())
    width = int(canvas.winfo_width())
    height = int(canvas.winfo_height())
    if width <= 1 or height <= 1:
        return None
    return (x0, y0, x0 + width, y0 + height)


def fit_size_to_bounds(size: Size, max_size: Size | None = None) -> Size:
    """Fit ``size`` into ``max_size`` without upscaling or changing aspect ratio."""
    width, height = int(size[0]), int(size[1])
    if max_size is None:
        return width, height
    max_width, max_height = int(max_size[0]), int(max_size[1])
    if max_width <= 0 or max_height <= 0:
        return width, height
    scale = min(1.0, max_width / max(1, width), max_height / max(1, height))
    if scale >= 1.0:
        return width, height
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def capture_human_frame(*, max_size: Size | None = (1600, 1200)) -> Image.Image | None:
    """Capture the active Pacman canvas as a PIL image.

    The default ``max_size`` is larger than all bundled layouts at zoom=1.0, so
    it does not affect normal captures.  It prevents huge GIF frames when users
    increase zoom or run on unusual high-DPI setups.
    """
    bbox = canvas_bbox()
    if bbox is None:
        return None
    try:
        frame = ImageGrab.grab(bbox=bbox)
    except Exception:
        return None

    target_size = fit_size_to_bounds(frame.size, max_size)
    if target_size != frame.size:
        frame = frame.resize(target_size, Image.Resampling.LANCZOS)
    return frame


def save_gif(frames: list[Image.Image], output_path: Path, frame_time: float) -> None:
    """Save captured frames as an animated GIF."""
    if not frames:
        return
    duration_ms = max(20, int(float(frame_time) * 1000))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        optimize=False,
        duration=duration_ms,
        loop=0,
    )
