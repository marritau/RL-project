"""Pure checks for visualization sizing helpers."""

from __future__ import annotations

from pacman_rldp.visuals.capture import expected_canvas_size, fit_size_to_bounds


def test_expected_small_classic_canvas_size_at_zoom_one() -> None:
    # smallClassic is 20x7 cells, grid=30px, plus one-cell border on each side
    # and a 35px info pane: width=(20+1)*30=630, height=(7+1)*30+35=275.
    assert expected_canvas_size(layout_width=20, layout_height=7, zoom=1.0) == (630, 275)


def test_fit_size_to_bounds_preserves_aspect_ratio_without_upscaling() -> None:
    assert fit_size_to_bounds((2520, 1680), (1600, 1200)) == (1600, 1067)
    assert fit_size_to_bounds((630, 275), (1600, 1200)) == (630, 275)
