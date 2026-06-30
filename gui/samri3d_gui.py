#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gui/samri3d_gui.py

Interactive Jupyter GUI for SAMRI-3D MRI segmentation — the 3D analogue of SAMRI v1's
ipywidgets demo. Built with ipywidgets + ipycanvas.

Workflow
--------
1. Choose a 3D volume (.nii.gz / .nii / .mha / .nrrd) and load it.
2. Navigate slices with the slider; pick the slice to prompt on.
3. Draw a bounding box (drag) and/or click positive/negative points on that slice.
4. Load a SAMRI-3D checkpoint (optionally GVT).
5. Click "Segment" — SAM2 propagates the prompt through the whole volume.
6. Toggle the overlay, scrub through the result, and save it as NIfTI.

Launch (inside the `samri` conda env, on a CUDA machine):

    from gui.samri3d_gui import launch
    launch()

Use **Jupyter Notebook** (ipycanvas mouse events are most reliable there); in JupyterLab
install the matching `ipycanvas`/`ipywidgets` labextensions first.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gui.inference_backend import (
    load_volume, load_predictor, run_prompt_inference, save_mask,
)
from test_vis3d_multi import normalize_to_uint8

# UI colors (match the paper / qualitative_study.py: yellow GT, magenta prediction)
_BOX_COLOR = "#FFFF00"        # prompt box
_POS_COLOR = "#00E5FF"        # positive points (cyan)
_NEG_COLOR = "#FF1744"        # negative points (red)
_PRED_COLOR = (233, 30, 99)   # magenta overlay (R,G,B)
_MAX_DISPLAY = 512            # largest canvas side in pixels

_DEFAULT_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"


class SAMRI3DGUI:
    """Stateful ipywidgets/ipycanvas application object. Call `.show()` to render."""

    def __init__(self, default_config: str = _DEFAULT_CONFIG, device: str = "cuda"):
        import ipywidgets as W
        from ipycanvas import MultiCanvas
        from ipyfilechooser import FileChooser

        self.W = W
        # ---- state ----
        self.vol0: Optional[np.ndarray] = None      # [S,H,W] view orientation
        self.ref_sitk = None
        self.view_axis = 0
        self.pred0: Optional[np.ndarray] = None     # [S,H,W] result
        self.predictor = None
        self.is_gvt = False
        self.device = device

        self.box: Optional[List[float]] = None      # [x0,y0,x1,y1] in image px
        self.points: List[Tuple[float, float]] = []
        self.point_labels: List[int] = []
        self._drag_start: Optional[Tuple[float, float]] = None
        self._scale = 1.0                           # canvas_px / image_px

        # ---- widgets: data + model ----
        self.fc_image = FileChooser(str(Path.home()))
        self.fc_image.title = "MRI volume"
        self.fc_image.filter_pattern = ["*.nii.gz", "*.nii", "*.mha", "*.mhd", "*.nrrd"]
        self.btn_load_img = W.Button(description="Load volume", icon="upload",
                                     button_style="info")
        self.dd_axis = W.Dropdown(options=[("auto (shortest)", -1), ("axis 0", 0),
                                           ("axis 1", 1), ("axis 2", 2)],
                                  value=-1, description="View axis")

        self.txt_config = W.Text(value=default_config, description="config")
        self.fc_ckpt = FileChooser(str(Path.home()))
        self.fc_ckpt.title = "Checkpoint (.pt/.pth)"
        self.fc_ckpt.filter_pattern = ["*.pt", "*.pth"]
        self.chk_gvt = W.Checkbox(value=False, description="GVT checkpoint")
        self.btn_load_model = W.Button(description="Load model", icon="cog",
                                       button_style="info")

        # ---- widgets: prompting ----
        self.sl_slice = W.IntSlider(value=0, min=0, max=0, description="slice",
                                    continuous_update=False)
        self.tg_mode = W.ToggleButtons(
            options=[("Box", "box"), ("+ Point", "pos"), ("- Point", "neg")],
            value="box", description="prompt")
        self.btn_clear = W.Button(description="Clear prompt", icon="eraser")
        self.btn_segment = W.Button(description="Segment", icon="play",
                                    button_style="success")

        # ---- widgets: result + save ----
        self.chk_overlay = W.Checkbox(value=True, description="show mask")
        self.sl_alpha = W.FloatSlider(value=0.4, min=0.0, max=1.0, step=0.05,
                                      description="opacity")
        self.txt_save = W.Text(value="", placeholder="/path/to/output_seg.nii.gz",
                               description="save to")
        self.btn_save = W.Button(description="Save mask", icon="save")

        self.status = W.Output(layout=W.Layout(border="1px solid #ddd",
                                               height="140px", overflow="auto"))

        # canvas: layer 0 = image, 1 = mask overlay, 2 = prompt/interaction
        self.canvas = MultiCanvas(3, width=256, height=256)
        self.canvas[2].on_mouse_down(self._on_down)
        self.canvas[2].on_mouse_move(self._on_move)
        self.canvas[2].on_mouse_up(self._on_up)

        # ---- wiring ----
        self.btn_load_img.on_click(lambda _: self._safe(self._load_image))
        self.btn_load_model.on_click(lambda _: self._safe(self._load_model))
        self.btn_segment.on_click(lambda _: self._safe(self._segment))
        self.btn_save.on_click(lambda _: self._safe(self._save))
        self.btn_clear.on_click(lambda _: self._safe(self._clear_prompt))
        self.sl_slice.observe(self._on_slice_change, names="value")
        self.chk_overlay.observe(lambda _: self._redraw_overlay(), names="value")
        self.sl_alpha.observe(lambda _: self._redraw_overlay(), names="value")

    # ------------------------------------------------------------------ logging
    def _log(self, msg: str) -> None:
        with self.status:
            print(msg)

    def _safe(self, fn) -> None:
        try:
            fn()
        except Exception as e:  # surface errors in the GUI rather than the kernel
            with self.status:
                print(f"[ERROR] {e}")
                traceback.print_exc()

    # ------------------------------------------------------------------ data
    def _load_image(self) -> None:
        path = self.fc_image.selected
        if not path:
            self._log("Select a volume first.")
            return
        axis = None if self.dd_axis.value == -1 else int(self.dd_axis.value)
        self.vol0, _, self.ref_sitk, self.view_axis = load_volume(path, axis)
        self.pred0 = None
        self._clear_prompt(redraw=False)

        S, H, W = self.vol0.shape
        self._scale = min(1.0, _MAX_DISPLAY / max(H, W))
        cw, ch = int(round(W * self._scale)), int(round(H * self._scale))
        for layer in self.canvas:
            layer.width, layer.height = cw, ch
        self.sl_slice.max = S - 1
        self.sl_slice.value = S // 2
        self._log(f"Loaded {Path(path).name}  shape(view)={self.vol0.shape}  "
                  f"view_axis={self.view_axis}")
        self._redraw_image()

    def _load_model(self) -> None:
        ckpt = self.fc_ckpt.selected
        if not ckpt:
            self._log("Select a checkpoint first.")
            return
        self._log(f"Loading model ({'GVT' if self.chk_gvt.value else 'plain'}) ... "
                  "this can take a moment.")
        self.predictor, self.is_gvt = load_predictor(
            self.txt_config.value, ckpt, device=self.device, use_gvt=self.chk_gvt.value)
        self._log(f"Model ready (GVT={self.is_gvt}, device={self.device}).")

    # ------------------------------------------------------------------ canvas math
    def _to_img(self, x: float, y: float) -> Tuple[float, float]:
        return x / self._scale, y / self._scale

    def _to_canvas(self, x: float, y: float) -> Tuple[float, float]:
        return x * self._scale, y * self._scale

    # ------------------------------------------------------------------ drawing
    def _redraw_image(self) -> None:
        if self.vol0 is None:
            return
        from PIL import Image
        sl = normalize_to_uint8(self.vol0[self.sl_slice.value])
        cw, ch = self.canvas[0].width, self.canvas[0].height
        rgb = np.asarray(Image.fromarray(sl).convert("RGB").resize((cw, ch)))
        self.canvas[0].put_image_data(rgb, 0, 0)
        self._redraw_overlay()
        self._redraw_prompt()

    def _redraw_overlay(self) -> None:
        layer = self.canvas[1]
        layer.clear()
        if self.pred0 is None or not self.chk_overlay.value:
            return
        from PIL import Image
        mask = (self.pred0[self.sl_slice.value] > 0).astype(np.uint8) * 255
        cw, ch = layer.width, layer.height
        m = np.asarray(Image.fromarray(mask).resize((cw, ch), Image.NEAREST)) > 127
        rgba = np.zeros((ch, cw, 4), dtype=np.uint8)
        rgba[m] = (*_PRED_COLOR, int(255 * self.sl_alpha.value))
        layer.put_image_data(rgba, 0, 0)

    def _redraw_prompt(self) -> None:
        layer = self.canvas[2]
        layer.clear()
        if self.box is not None:
            x0, y0, x1, y1 = self.box
            cx0, cy0 = self._to_canvas(x0, y0)
            cx1, cy1 = self._to_canvas(x1, y1)
            layer.line_width = 2
            layer.stroke_style = _BOX_COLOR
            layer.stroke_rect(min(cx0, cx1), min(cy0, cy1),
                              abs(cx1 - cx0), abs(cy1 - cy0))
        for (px, py), lab in zip(self.points, self.point_labels):
            cx, cy = self._to_canvas(px, py)
            layer.fill_style = _POS_COLOR if lab == 1 else _NEG_COLOR
            layer.fill_arc(cx, cy, 4, 0, 2 * np.pi)

    # ------------------------------------------------------------------ mouse
    def _on_down(self, x, y) -> None:
        if self.vol0 is None:
            return
        ix, iy = self._to_img(x, y)
        if self.tg_mode.value == "box":
            self._drag_start = (ix, iy)
            self.box = [ix, iy, ix, iy]
        else:
            self.points.append((ix, iy))
            self.point_labels.append(1 if self.tg_mode.value == "pos" else 0)
            self._redraw_prompt()

    def _on_move(self, x, y) -> None:
        if self._drag_start is None or self.tg_mode.value != "box":
            return
        ix, iy = self._to_img(x, y)
        x0, y0 = self._drag_start
        self.box = [x0, y0, ix, iy]
        self._redraw_prompt()

    def _on_up(self, x, y) -> None:
        if self._drag_start is None:
            return
        ix, iy = self._to_img(x, y)
        x0, y0 = self._drag_start
        # normalize so x0<x1, y0<y1; ignore degenerate clicks
        self.box = [min(x0, ix), min(y0, iy), max(x0, ix), max(y0, iy)]
        if (self.box[2] - self.box[0]) < 2 or (self.box[3] - self.box[1]) < 2:
            self.box = None
        self._drag_start = None
        self._redraw_prompt()

    def _clear_prompt(self, redraw: bool = True) -> None:
        self.box = None
        self.points, self.point_labels = [], []
        self._drag_start = None
        if redraw:
            self._redraw_prompt()

    def _on_slice_change(self, _change) -> None:
        self._redraw_image()

    # ------------------------------------------------------------------ inference
    def _segment(self) -> None:
        if self.vol0 is None:
            self._log("Load a volume first.")
            return
        if self.predictor is None:
            self._log("Load a model first.")
            return
        if self.box is None and not self.points:
            self._log("Draw a box and/or add points on the current slice first.")
            return
        box = np.asarray(self.box, dtype=np.float32) if self.box else None
        pts = np.asarray(self.points, dtype=np.float32) if self.points else None
        labs = np.asarray(self.point_labels, dtype=np.int32) if self.points else None
        self._log(f"Segmenting from slice {self.sl_slice.value} "
                  f"(box={'yes' if box is not None else 'no'}, points={len(self.points)}) ...")
        self.pred0 = run_prompt_inference(
            self.predictor, self.vol0, self.sl_slice.value,
            box=box, points=pts, labels=labs, use_gvt=self.is_gvt)
        vox = int((self.pred0 > 0).sum())
        self._log(f"Done. Predicted foreground voxels: {vox:,}.")
        self.chk_overlay.value = True
        self._redraw_overlay()

    def _save(self) -> None:
        if self.pred0 is None:
            self._log("Nothing to save — run Segment first.")
            return
        out = self.txt_save.value.strip()
        if not out:
            self._log("Enter an output path in 'save to'.")
            return
        save_mask(self.pred0, self.view_axis, self.ref_sitk, out)
        self._log(f"Saved mask -> {out}")

    # ------------------------------------------------------------------ layout
    def show(self):
        W = self.W
        data_box = W.VBox([W.HTML("<b>1 · Data</b>"), self.fc_image,
                           W.HBox([self.btn_load_img, self.dd_axis])])
        model_box = W.VBox([W.HTML("<b>2 · Model</b>"), self.txt_config, self.fc_ckpt,
                            W.HBox([self.chk_gvt, self.btn_load_model])])
        prompt_box = W.VBox([W.HTML("<b>3 · Prompt &amp; segment</b>"),
                             self.sl_slice, self.tg_mode,
                             W.HBox([self.btn_clear, self.btn_segment]),
                             W.HBox([self.chk_overlay, self.sl_alpha]),
                             W.HBox([self.txt_save, self.btn_save])])
        controls = W.VBox([data_box, W.HTML("<hr>"), model_box, W.HTML("<hr>"),
                           prompt_box],
                          layout=W.Layout(width="380px"))
        return W.VBox([
            W.HTML("<h3>SAMRI-3D — interactive 3D MRI segmentation</h3>"),
            W.HBox([controls, self.canvas]),
            W.HTML("<b>Log</b>"), self.status,
        ])


def launch(default_config: str = _DEFAULT_CONFIG, device: str = "cuda"):
    """Build the GUI and display it (call from a Jupyter cell)."""
    from IPython.display import display
    app = SAMRI3DGUI(default_config=default_config, device=device)
    display(app.show())
    return app
