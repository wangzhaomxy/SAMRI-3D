"""SAMRI-3D interactive GUI (ipywidgets + ipycanvas).

Usage (in a Jupyter cell, inside the `samri` env on a CUDA machine):

    from gui import launch
    launch()
"""

from gui.samri3d_gui import launch, SAMRI3DGUI

__all__ = ["launch", "SAMRI3DGUI"]
