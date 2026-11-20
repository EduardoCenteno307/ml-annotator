#!/usr/bin/env python3
"""
ml-annotator - a lightweight Tkinter annotation tool.

Annotation types : bounding boxes (detection) and polygons (segmentation)
Export formats   : YOLO (detect / segment), COCO JSON, Pascal VOC XML
Project storage  : annotations are auto-saved next to your images in
                   ``annotations_project.json`` and reloaded on next open.
"""

from __future__ import annotations

import copy
import json
import math
import os
import re
import sys
import xml.etree.ElementTree as ET
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog

from PIL import Image, ImageOps, ImageTk

# ── Constants ────────────────────────────────────────────────────────────────
APP_NAME = "ml-annotator"
PROJECT_FILE = "annotations_project.json"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

COLORS = [
    "#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6",
    "#1ABC9C", "#E67E22", "#ECF0F1", "#E91E63", "#00BCD4",
]
BG, BG_DARK, BG_PANEL = "#1E1E2E", "#11111B", "#181825"
BTN_BG, FG, FG_DIM, ACCENT = "#313244", "#CDD6F4", "#6C7086", "#7C3AED"
MONO = "Courier"

HANDLE_RADIUS = 5
HANDLE_HIT = 9          # px on screen
SNAP_RADIUS = 10        # px on screen, polygon close
MIN_BOX_SIZE = 5        # image px
MIN_SCALE, MAX_SCALE = 0.02, 40.0
UNDO_LIMIT = 100
CMD = "Command" if sys.platform == "darwin" else "Control"


# ── Annotation helpers (annotations are plain JSON-serialisable dicts) ───────
def make_bbox(x1, y1, x2, y2, label):
    return {"type": "bbox", "label": label,
            "x1": min(x1, x2), "y1": min(y1, y2),
            "x2": max(x1, x2), "y2": max(y1, y2)}


def make_polygon(points, label):
    return {"type": "polygon", "label": label, "points": [list(p) for p in points]}


def ann_points(ann):
    """Vertices of an annotation (a bbox is returned as its 4 corners TL,TR,BR,BL)."""
    if ann["type"] == "bbox":
        return [(ann["x1"], ann["y1"]), (ann["x2"], ann["y1"]),
                (ann["x2"], ann["y2"]), (ann["x1"], ann["y2"])]
    return [tuple(p) for p in ann["points"]]


def ann_bounds(ann):
    xs, ys = zip(*ann_points(ann))
    return min(xs), min(ys), max(xs), max(ys)


def polygon_area(pts):
    """Shoelace formula."""
    n = len(pts)
    s = 0.0
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def ann_area(ann):
    if ann["type"] == "bbox":
        return (ann["x2"] - ann["x1"]) * (ann["y2"] - ann["y1"])
    return polygon_area(ann_points(ann))


def point_in_polygon(x, y, pts):
    inside = False
    j = len(pts) - 1
    for i in range(len(pts)):
        xi, yi = pts[i]
        xj, yj = pts[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def ann_contains(ann, x, y):
    if ann["type"] == "bbox":
        return ann["x1"] <= x <= ann["x2"] and ann["y1"] <= y <= ann["y2"]
    return point_in_polygon(x, y, ann_points(ann))


def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def clamp01(v):
    return min(max(v, 0.0), 1.0)


# ── Class picker dialog ──────────────────────────────────────────────────────
class ClassDialog(tk.Toplevel):
    """Modal dialog: pick an existing class or type a new one."""

    def __init__(self, parent, classes, color_fn, default=None, title="Assign Class"):
        super().__init__(parent)
        self.result = None
        self.classes = list(classes)
        self.title(title)
        self.configure(bg=BG_PANEL)
        self.resizable(False, False)
        self.transient(parent)

        tk.Label(self, text=title, bg=BG_PANEL, fg="#CBA6F7",
                 font=(MONO, 12, "bold")).pack(pady=(12, 4))

        self.lb = None
        if self.classes:
            tk.Label(self, text="Pick existing:", bg=BG_PANEL, fg="#A6ADC8",
                     font=(MONO, 9)).pack(anchor=tk.W, padx=12)
            frame = tk.Frame(self, bg=BG_PANEL)
            frame.pack(fill=tk.X, padx=12, pady=(2, 8))
            sb = tk.Scrollbar(frame, orient=tk.VERTICAL)
            sb.pack(side=tk.RIGHT, fill=tk.Y)
            self.lb = tk.Listbox(frame, bg=BTN_BG, fg=FG, selectbackground=ACCENT,
                                 font=(MONO, 10), height=min(7, len(self.classes)),
                                 bd=0, highlightthickness=0, exportselection=False,
                                 yscrollcommand=sb.set)
            self.lb.pack(side=tk.LEFT, fill=tk.X, expand=True)
            sb.config(command=self.lb.yview)
            for i, c in enumerate(self.classes):
                self.lb.insert(tk.END, f"  {c}")
                self.lb.itemconfig(i, fg=color_fn(c))
            if default in self.classes:
                idx = self.classes.index(default)
                self.lb.selection_set(idx)
                self.lb.see(idx)
            self.lb.bind("<<ListboxSelect>>", lambda e: self.entry_var.set(""))
            self.lb.bind("<Double-Button-1>", lambda e: self._confirm())

        tk.Label(self, text="Or type a new class:", bg=BG_PANEL, fg="#A6ADC8",
                 font=(MONO, 9)).pack(anchor=tk.W, padx=12)
        self.entry_var = tk.StringVar()
        entry = tk.Entry(self, textvariable=self.entry_var, bg=BTN_BG, fg=FG,
                         insertbackground=FG, font=(MONO, 11), bd=0,
                         highlightthickness=1, highlightcolor=ACCENT,
                         highlightbackground="#45475A")
        entry.pack(fill=tk.X, padx=12, pady=(2, 12))
        entry.bind("<Return>", lambda e: self._confirm())
        entry.bind("<KeyPress>", self._on_entry_key)

        btns = tk.Frame(self, bg=BG_PANEL)
        btns.pack(fill=tk.X, padx=12, pady=(0, 12))
        tk.Button(btns, text="✓ Confirm", command=self._confirm, bg=ACCENT, fg=FG,
                  font=(MONO, 10), relief=tk.FLAT, cursor="hand2"
                  ).pack(side=tk.LEFT, expand=True, fill=tk.X)
        tk.Button(btns, text="✕ Cancel", command=self.destroy, bg="#45475A", fg="#F38BA8",
                  font=(MONO, 10), relief=tk.FLAT, cursor="hand2"
                  ).pack(side=tk.LEFT, padx=(4, 0))
        self.bind("<Escape>", lambda e: self.destroy())

        self.update_idletasks()
        w, h = 300, self.winfo_reqheight()
        x = parent.winfo_rootx() + (parent.winfo_width() - w) // 2
        y = parent.winfo_rooty() + (parent.winfo_height() - h) // 2
        self.geometry(f"{w}x{h}+{max(x, 0)}+{max(y, 0)}")
        self.grab_set()
        entry.focus_set()
        self.wait_window(self)

    def _on_entry_key(self, event):
        # typing a new name overrides any list selection
        if self.lb is not None and event.char and event.char.isprintable():
            self.lb.selection_clear(0, tk.END)

    def _confirm(self):
        text = self.entry_var.get().strip()
        if text:
            self.result = text
        elif self.lb is not None and self.lb.curselection():
            self.result = self.classes[self.lb.curselection()[0]]
        else:
            return  # nothing chosen: keep dialog open
        self.destroy()


# ── Main application ─────────────────────────────────────────────────────────
class AnnotatorApp(tk.Tk):
    def __init__(self, initial_folder=None):
        super().__init__()
        self.title(APP_NAME)
        self.geometry("1280x800")
        self.minsize(900, 600)
        self.configure(bg=BG)

        # Dataset state
        self.folder = None
        self.image_paths = []
        self.current_index = -1
        self.annotations = {}            # {abs_path: [annotation, ...]}
        self.classes = []                # derived from annotations (stable order)
        self.last_label = None
        self._size_cache = {}
        self.dirty = False

        # View state
        self.pil_image = None
        self.tk_image = None
        self.scale = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.tool = tk.StringVar(value="bbox")   # "select" | "bbox" | "polygon"

        # Drawing state
        self.drawing = False
        self.start_x = self.start_y = 0.0
        self.current_rect = None
        self.polygon_points = []

        # Selection / edit state
        self.selected_idx = None
        self.edit_mode = None            # ("move", None) | ("handle", index)
        self.edit_origin = (0, 0)
        self.edit_orig = None
        self._pending_snap = None
        self._edit_moved = False

        # Pan state
        self._pan_start = (0, 0)
        self._pan_off = (0, 0)
        self._pan_moved = False

        # History
        self.undo_stack = []
        self.redo_stack = []

        self._build_ui()
        self._bind_events()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        if initial_folder and os.path.isdir(initial_folder):
            self.after(150, lambda: self.load_folder(initial_folder))

    # ── UI layout ────────────────────────────────────────────────────────────
    def _build_ui(self):
        toolbar = tk.Frame(self, bg=BG_PANEL, pady=4)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        tk.Label(toolbar, text="🏷 ml-annotator", bg=BG_PANEL, fg="#CBA6F7",
                 font=(MONO, 14, "bold")).pack(side=tk.LEFT, padx=12)

        tool_frame = tk.LabelFrame(toolbar, text="Tool", bg=BG_PANEL, fg="#A6ADC8",
                                   font=(MONO, 9), bd=1, relief=tk.GROOVE)
        tool_frame.pack(side=tk.LEFT, padx=8)
        for text, val in [("➤ Select (V)", "select"), ("⬜ BBox (B)", "bbox"),
                          ("⬡ Polygon (P)", "polygon")]:
            tk.Radiobutton(tool_frame, text=text, variable=self.tool, value=val,
                           bg=BG_PANEL, fg=FG, selectcolor=BTN_BG,
                           activebackground=BG_PANEL, activeforeground=FG,
                           font=(MONO, 10), command=self._on_tool_change
                           ).pack(side=tk.LEFT, padx=6, pady=2)

        btn = {"bg": BTN_BG, "fg": FG, "font": (MONO, 10), "relief": tk.FLAT,
               "padx": 8, "pady": 4, "cursor": "hand2", "activebackground": ACCENT,
               "activeforeground": FG}
        tk.Button(toolbar, text="📂 Open Folder", command=self.open_folder, **btn
                  ).pack(side=tk.LEFT, padx=4)
        tk.Button(toolbar, text="💾 Save", command=lambda: self.save_project(True), **btn
                  ).pack(side=tk.LEFT, padx=2)

        mb = tk.Menubutton(toolbar, text="📤 Export ▾", **btn)
        menu = tk.Menu(mb, tearoff=0, bg=BTN_BG, fg=FG, activebackground=ACCENT,
                       font=(MONO, 10))
        menu.add_command(label="YOLO – detection (boxes)", command=lambda: self.export("yolo_detect"))
        menu.add_command(label="YOLO – segmentation (polygons)", command=lambda: self.export("yolo_segment"))
        menu.add_command(label="COCO JSON", command=lambda: self.export("coco"))
        menu.add_command(label="Pascal VOC XML", command=lambda: self.export("voc"))
        mb.config(menu=menu)
        mb.pack(side=tk.LEFT, padx=2)

        tk.Button(toolbar, text="⟳ Fit", command=self.fit_image, **btn).pack(side=tk.LEFT, padx=4)
        tk.Button(toolbar, text="↶", command=self.undo, **btn).pack(side=tk.LEFT, padx=(8, 0))
        tk.Button(toolbar, text="↷", command=self.redo, **btn).pack(side=tk.LEFT, padx=2)

        self.counter_label = tk.Label(toolbar, text="No images", bg=BG_PANEL,
                                      fg="#89B4FA", font=(MONO, 10))
        self.counter_label.pack(side=tk.RIGHT, padx=12)

        # Status bar (packed before main so it is never pushed off-screen)
        status_bar = tk.Frame(self, bg=BG_PANEL)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_var = tk.StringVar(value="Open a folder to begin (Ctrl+O).")
        tk.Label(status_bar, textvariable=self.status_var, bg=BG_PANEL, fg="#89DCEB",
                 font=(MONO, 9), anchor=tk.W, padx=8).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.coord_var = tk.StringVar(value="")
        tk.Label(status_bar, textvariable=self.coord_var, bg=BG_PANEL, fg=FG_DIM,
                 font=(MONO, 9), padx=8).pack(side=tk.RIGHT)

        main = tk.Frame(self, bg=BG)
        main.pack(fill=tk.BOTH, expand=True)

        # Left panel: images + classes
        left = tk.Frame(main, bg=BG_PANEL, width=260)
        left.pack(side=tk.LEFT, fill=tk.Y)
        left.pack_propagate(False)

        tk.Label(left, text="Images", bg=BG_PANEL, fg="#CBA6F7",
                 font=(MONO, 11, "bold")).pack(pady=(8, 2))
        frame = tk.Frame(left, bg=BG_PANEL)
        frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)
        vs = tk.Scrollbar(frame, orient=tk.VERTICAL)
        hs = tk.Scrollbar(frame, orient=tk.HORIZONTAL)
        vs.pack(side=tk.RIGHT, fill=tk.Y)
        hs.pack(side=tk.BOTTOM, fill=tk.X)
        self.img_listbox = tk.Listbox(frame, bg=BTN_BG, fg=FG, selectbackground=ACCENT,
                                      font=(MONO, 9), bd=0, highlightthickness=0,
                                      exportselection=False,
                                      yscrollcommand=vs.set, xscrollcommand=hs.set)
        self.img_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vs.config(command=self.img_listbox.yview)
        hs.config(command=self.img_listbox.xview)
        self.img_listbox.bind("<<ListboxSelect>>", self._on_image_select)

        hdr = tk.Frame(left, bg=BG_PANEL)
        hdr.pack(fill=tk.X, padx=6, pady=(8, 2))
        tk.Label(hdr, text="Classes", bg=BG_PANEL, fg="#CBA6F7",
                 font=(MONO, 11, "bold")).pack(side=tk.LEFT)
        self.cls_count_label = tk.Label(hdr, text="", bg=BG_PANEL, fg=FG_DIM, font=(MONO, 8))
        self.cls_count_label.pack(side=tk.RIGHT)
        self.class_listbox = tk.Listbox(left, bg=BTN_BG, fg=FG, selectbackground="#45475A",
                                        font=(MONO, 9), height=8, bd=0,
                                        highlightthickness=0, exportselection=False)
        self.class_listbox.pack(fill=tk.X, padx=6)
        self.class_listbox.bind("<Double-Button-1>", self._rename_class)
        tk.Label(left, text="auto-detected · double-click to rename", bg=BG_PANEL,
                 fg=FG_DIM, font=(MONO, 7)).pack(padx=6, pady=(0, 6), anchor=tk.W)

        # Right panel: annotations of current image
        right = tk.Frame(main, bg=BG_PANEL, width=220)
        right.pack(side=tk.RIGHT, fill=tk.Y)
        right.pack_propagate(False)
        tk.Label(right, text="Annotations", bg=BG_PANEL, fg="#CBA6F7",
                 font=(MONO, 11, "bold")).pack(pady=(8, 2))
        aframe = tk.Frame(right, bg=BG_PANEL)
        aframe.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)
        asb = tk.Scrollbar(aframe, orient=tk.VERTICAL)
        asb.pack(side=tk.RIGHT, fill=tk.Y)
        self.ann_listbox = tk.Listbox(aframe, bg=BTN_BG, fg=FG, selectbackground=ACCENT,
                                      font=(MONO, 9), bd=0, highlightthickness=0,
                                      exportselection=False, yscrollcommand=asb.set)
        self.ann_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        asb.config(command=self.ann_listbox.yview)
        self.ann_listbox.bind("<<ListboxSelect>>", self._on_ann_select)

        abtns = tk.Frame(right, bg=BG_PANEL)
        abtns.pack(fill=tk.X, padx=6, pady=4)
        tk.Button(abtns, text="🗑 Delete", command=self.delete_annotation, bg="#45475A",
                  fg="#F38BA8", font=(MONO, 9), relief=tk.FLAT, cursor="hand2"
                  ).pack(side=tk.LEFT, expand=True, fill=tk.X)
        tk.Button(abtns, text="✏ Relabel", command=self.relabel_annotation, bg="#45475A",
                  fg="#A6E3A1", font=(MONO, 9), relief=tk.FLAT, cursor="hand2"
                  ).pack(side=tk.LEFT, padx=(2, 0), expand=True, fill=tk.X)

        # Center: canvas + navigation
        center = tk.Frame(main, bg=BG_DARK)
        center.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        nav = tk.Frame(center, bg=BG_PANEL, pady=4)
        nav.pack(side=tk.BOTTOM, fill=tk.X)
        nb = {"bg": BTN_BG, "fg": FG, "font": (MONO, 11), "relief": tk.FLAT,
              "padx": 12, "pady": 4, "cursor": "hand2"}
        tk.Button(nav, text="◀ Prev", command=self.prev_image, **nb).pack(side=tk.LEFT, padx=8)
        self.nav_label = tk.Label(nav, text="", bg=BG_PANEL, fg="#A6ADC8", font=(MONO, 10))
        self.nav_label.pack(side=tk.LEFT, expand=True)
        tk.Button(nav, text="Next ▶", command=self.next_image, **nb).pack(side=tk.RIGHT, padx=8)

        inner = tk.Frame(center, bg=BG_DARK)
        inner.pack(fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(inner, bg=BG_DARK, cursor="crosshair", bd=0,
                                highlightthickness=0)
        self.canvas_hscroll = tk.Scrollbar(inner, orient=tk.HORIZONTAL, command=self._canvas_xview)
        self.canvas_vscroll = tk.Scrollbar(inner, orient=tk.VERTICAL, command=self._canvas_yview)
        self.canvas_hscroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas_vscroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def _bind_events(self):
        c = self.canvas
        c.bind("<ButtonPress-1>", self._on_mouse_press)
        c.bind("<B1-Motion>", self._on_mouse_drag)
        c.bind("<ButtonRelease-1>", self._on_mouse_release)
        c.bind("<Motion>", self._on_mouse_motion)
        c.bind("<Double-Button-1>", self._on_double_click)
        c.bind("<Configure>", lambda e: self.redraw())
        # Zoom (Windows/macOS use <MouseWheel>, Linux uses Button-4/5)
        c.bind("<MouseWheel>", lambda e: self._zoom(1.1 if e.delta > 0 else 1 / 1.1, e.x, e.y))
        c.bind("<Button-4>", lambda e: self._zoom(1.1, e.x, e.y))
        c.bind("<Button-5>", lambda e: self._zoom(1 / 1.1, e.x, e.y))
        # Pan with right / middle button drag; a plain right-click cancels / deselects
        for b in ("2", "3"):
            c.bind(f"<ButtonPress-{b}>", self._on_pan_start)
            c.bind(f"<B{b}-Motion>", self._on_pan_drag)
            c.bind(f"<ButtonRelease-{b}>", self._on_pan_end)

        self.bind("<Delete>", lambda e: self.delete_annotation())
        self.bind("<BackSpace>", lambda e: self._polygon_undo_point())
        self.bind("<Escape>", lambda e: self._cancel_polygon())
        self.bind("<Return>", lambda e: self._finish_polygon())
        self.bind("<Left>", lambda e: self.prev_image())
        self.bind("<Right>", lambda e: self.next_image())
        for key, tool in (("v", "select"), ("b", "bbox"), ("p", "polygon")):
            self.bind(f"<KeyPress-{key}>", lambda e, t=tool: self._set_tool(t))
            self.bind(f"<KeyPress-{key.upper()}>", lambda e, t=tool: self._set_tool(t))
        self.bind(f"<{CMD}-o>", lambda e: self.open_folder())
        self.bind(f"<{CMD}-s>", lambda e: self.save_project(True))
        self.bind(f"<{CMD}-z>", lambda e: self.undo())
        self.bind(f"<{CMD}-y>", lambda e: self.redo())
        self.bind(f"<{CMD}-Shift-Z>", lambda e: self.redo())

    # ── Helpers ──────────────────────────────────────────────────────────────
    def _path(self):
        if 0 <= self.current_index < len(self.image_paths):
            return self.image_paths[self.current_index]
        return None

    def _anns(self):
        p = self._path()
        return self.annotations.setdefault(p, []) if p else []

    def _selected_ann(self):
        anns = self._anns()
        if self.selected_idx is not None and 0 <= self.selected_idx < len(anns):
            return anns[self.selected_idx]
        return None

    def _image_size(self, path):
        """Size after EXIF orientation, cached, without decoding the full image."""
        if path not in self._size_cache:
            with Image.open(path) as im:
                w, h = im.size
                try:
                    if im.getexif().get(274, 1) in (5, 6, 7, 8):
                        w, h = h, w
                except Exception:
                    pass
            self._size_cache[path] = (w, h)
        return self._size_cache[path]

    def _clamp_img(self, ix, iy):
        iw, ih = self.pil_image.size
        return min(max(ix, 0), iw), min(max(iy, 0), ih)

    def canvas_to_img(self, cx, cy):
        return (cx - self.offset_x) / self.scale, (cy - self.offset_y) / self.scale

    def img_to_canvas(self, ix, iy):
        return ix * self.scale + self.offset_x, iy * self.scale + self.offset_y

    def _color_for_label(self, label):
        idx = self.classes.index(label) if label in self.classes else len(self.classes)
        return COLORS[idx % len(COLORS)]

    def _current_color(self):
        return self._color_for_label(self.last_label) if self.last_label else COLORS[len(self.classes) % len(COLORS)]

    def _set_tool(self, tool):
        self.tool.set(tool)
        self._on_tool_change()

    # ── History (undo / redo) ────────────────────────────────────────────────
    def _snapshot(self):
        p = self._path()
        return (p, copy.deepcopy(self.annotations.get(p, [])), self.selected_idx)

    def _commit(self, snap):
        self.undo_stack.append(snap)
        del self.undo_stack[:-UNDO_LIMIT]
        self.redo_stack.clear()

    def _push_undo(self):
        if self._path():
            self._commit(self._snapshot())

    def _restore(self, snap):
        path, anns, sel = snap
        if path not in self.image_paths:
            return
        idx = self.image_paths.index(path)
        if idx != self.current_index:
            self.load_image(idx)
        self.annotations[path] = copy.deepcopy(anns)
        self.selected_idx = sel
        self._after_change()

    def undo(self):
        if self.polygon_points or not self.undo_stack:
            return
        self.redo_stack.append(self._snapshot())
        self._restore(self.undo_stack.pop())
        self.status_var.set("Undo")

    def redo(self):
        if self.polygon_points or not self.redo_stack:
            return
        self.undo_stack.append(self._snapshot())
        self._restore(self.redo_stack.pop())
        self.status_var.set("Redo")

    def _after_change(self):
        self._rebuild_classes()
        self._refresh_ann_list()
        self._update_img_marker(self.current_index)
        self.redraw()
        self.dirty = True

    # ── Folder / project I/O ─────────────────────────────────────────────────
    def open_folder(self):
        folder = filedialog.askdirectory(title="Select image folder")
        if folder:
            self.load_folder(folder)

    def load_folder(self, folder):
        self.save_project()
        paths = sorted((os.path.join(folder, f) for f in os.listdir(folder)
                        if os.path.splitext(f)[1].lower() in IMAGE_EXTS),
                       key=lambda p: natural_key(os.path.basename(p)))
        if not paths:
            messagebox.showinfo("No images", "No supported images found in that folder.")
            return
        self.folder = folder
        self.image_paths = paths
        self.annotations = {}
        self.classes = []
        self._size_cache.clear()
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.current_index = -1
        self.dirty = False
        self._load_project()
        self._rebuild_classes()
        self._refresh_img_list()
        self.update_idletasks()
        self.load_image(0)
        n_ann = sum(len(a) for a in self.annotations.values())
        self.title(f"{APP_NAME} — {folder}")
        self.status_var.set(f"Loaded {len(paths)} images ({n_ann} annotations restored) from {folder}")

    def _project_path(self):
        return os.path.join(self.folder, PROJECT_FILE) if self.folder else None

    def save_project(self, verbose=False):
        if not self.folder or (not self.dirty and not verbose):
            return
        data = {
            "version": 1,
            "classes": self.classes,
            "images": {os.path.relpath(p, self.folder): anns
                       for p, anns in self.annotations.items() if anns},
        }
        target = self._project_path()
        tmp = target + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, target)
            self.dirty = False
            if verbose:
                self.status_var.set(f"Saved → {target}")
        except OSError as exc:
            messagebox.showerror("Save failed", str(exc))

    def _load_project(self):
        target = self._project_path()
        if not target or not os.path.isfile(target):
            return
        try:
            with open(target, encoding="utf-8") as f:
                data = json.load(f)
            known = {os.path.relpath(p, self.folder): p for p in self.image_paths}
            for rel, anns in data.get("images", {}).items():
                if rel in known:
                    self.annotations[known[rel]] = anns
            self.classes = [c for c in data.get("classes", []) if isinstance(c, str)]
        except (OSError, ValueError, AttributeError) as exc:
            messagebox.showwarning("Project not loaded",
                                   f"Could not read {PROJECT_FILE}:\n{exc}")

    def _on_close(self):
        self.save_project()
        self.destroy()

    # ── Image list / navigation ──────────────────────────────────────────────
    def _img_item_text(self, p):
        return ("✔ " if self.annotations.get(p) else "  ") + os.path.basename(p)

    def _refresh_img_list(self):
        self.img_listbox.delete(0, tk.END)
        for i, p in enumerate(self.image_paths):
            self.img_listbox.insert(tk.END, self._img_item_text(p))
            self.img_listbox.itemconfig(i, fg="#A6E3A1" if self.annotations.get(p) else FG)
        self._sync_img_selection()

    def _update_img_marker(self, i):
        if not (0 <= i < len(self.image_paths)):
            return
        p = self.image_paths[i]
        self.img_listbox.delete(i)
        self.img_listbox.insert(i, self._img_item_text(p))
        self.img_listbox.itemconfig(i, fg="#A6E3A1" if self.annotations.get(p) else FG)
        self._sync_img_selection()

    def _sync_img_selection(self):
        self.img_listbox.selection_clear(0, tk.END)
        if 0 <= self.current_index < len(self.image_paths):
            self.img_listbox.selection_set(self.current_index)
            self.img_listbox.see(self.current_index)

    def load_image(self, idx):
        if not (0 <= idx < len(self.image_paths)):
            return
        path = self.image_paths[idx]
        try:
            img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        except Exception as exc:
            messagebox.showerror("Cannot open image", f"{path}\n\n{exc}")
            return
        self.save_project()
        self._cancel_polygon(redraw=False)
        self.drawing = False
        self.edit_mode = None
        self.pil_image = img
        self.current_index = idx
        self._size_cache[path] = img.size
        self.annotations.setdefault(path, [])
        self.selected_idx = None
        self.fit_image()
        self._refresh_ann_list()
        self._sync_img_selection()
        self.nav_label.config(text=f"{idx + 1} / {len(self.image_paths)}")
        self.counter_label.config(text=os.path.basename(path))
        self.status_var.set(f"{path}  ({img.width}×{img.height})")

    def prev_image(self):
        if self.current_index > 0:
            self.load_image(self.current_index - 1)

    def next_image(self):
        if self.current_index < len(self.image_paths) - 1:
            self.load_image(self.current_index + 1)

    def _on_image_select(self, _event):
        sel = self.img_listbox.curselection()
        if sel and sel[0] != self.current_index:
            self.load_image(sel[0])

    # ── View: fit / zoom / pan / scroll ──────────────────────────────────────
    def fit_image(self):
        if self.pil_image is None:
            return
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw <= 1 or ch <= 1:
            cw, ch = 800, 600
        iw, ih = self.pil_image.size
        self.scale = max(MIN_SCALE, min(cw / iw, ch / ih) * 0.95)
        self.offset_x = (cw - iw * self.scale) / 2
        self.offset_y = (ch - ih * self.scale) / 2
        self.redraw()

    def _zoom(self, factor, cx, cy):
        if self.pil_image is None:
            return
        new_scale = min(max(self.scale * factor, MIN_SCALE), MAX_SCALE)
        factor = new_scale / self.scale
        self.offset_x = cx - (cx - self.offset_x) * factor   # zoom around the cursor
        self.offset_y = cy - (cy - self.offset_y) * factor
        self.scale = new_scale
        self._clamp_view()
        self.redraw()

    def _clamp_view(self):
        """Keep at least a margin of the image visible."""
        if self.pil_image is None:
            return
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        iw, ih = self.pil_image.width * self.scale, self.pil_image.height * self.scale
        m = 60
        self.offset_x = min(max(self.offset_x, m - iw), cw - m)
        self.offset_y = min(max(self.offset_y, m - ih), ch - m)

    def _canvas_xview(self, *args):
        if self.pil_image is None:
            return
        iw = self.pil_image.width * self.scale
        cw = self.canvas.winfo_width()
        if args[0] == "moveto":
            self.offset_x = -float(args[1]) * iw
        elif args[0] == "scroll":
            step = cw * (0.9 if args[2] == "pages" else 0.1)
            self.offset_x -= int(args[1]) * step
        self._clamp_view()
        self.redraw()

    def _canvas_yview(self, *args):
        if self.pil_image is None:
            return
        ih = self.pil_image.height * self.scale
        ch = self.canvas.winfo_height()
        if args[0] == "moveto":
            self.offset_y = -float(args[1]) * ih
        elif args[0] == "scroll":
            step = ch * (0.9 if args[2] == "pages" else 0.1)
            self.offset_y -= int(args[1]) * step
        self._clamp_view()
        self.redraw()

    def _update_scrollbars(self):
        if self.pil_image is None:
            return
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        iw, ih = self.pil_image.width * self.scale, self.pil_image.height * self.scale
        for bar, off, view, total in ((self.canvas_hscroll, self.offset_x, cw, iw),
                                      (self.canvas_vscroll, self.offset_y, ch, ih)):
            if total <= view and off >= 0 and off + total <= view:
                bar.set(0.0, 1.0)
            else:
                lo = max(0.0, -off / total)
                hi = min(1.0, (view - off) / total)
                bar.set(lo, max(hi, lo + 0.01))

    def _on_pan_start(self, event):
        self._pan_start = (event.x, event.y)
        self._pan_off = (self.offset_x, self.offset_y)
        self._pan_moved = False

    def _on_pan_drag(self, event):
        dx, dy = event.x - self._pan_start[0], event.y - self._pan_start[1]
        if not self._pan_moved and math.hypot(dx, dy) < 4:
            return
        self._pan_moved = True
        self.canvas.config(cursor="fleur")
        self.offset_x = self._pan_off[0] + dx
        self.offset_y = self._pan_off[1] + dy
        self._clamp_view()
        self.redraw()

    def _on_pan_end(self, _event):
        self.canvas.config(cursor="crosshair")
        if not self._pan_moved:          # plain right-click
            if self.polygon_points:
                self._cancel_polygon()
            elif self.selected_idx is not None:
                self.selected_idx = None
                self._refresh_ann_list()
                self.redraw()

    # ── Rendering ────────────────────────────────────────────────────────────
    def redraw(self):
        self.canvas.delete("all")
        if self.pil_image is None:
            return
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        iw, ih = self.pil_image.size
        # Only resize the visible part of the image: fast and memory-safe at any zoom.
        x0 = max(0, int(math.floor(-self.offset_x / self.scale)))
        y0 = max(0, int(math.floor(-self.offset_y / self.scale)))
        x1 = min(iw, int(math.ceil((cw - self.offset_x) / self.scale)))
        y1 = min(ih, int(math.ceil((ch - self.offset_y) / self.scale)))
        if x1 > x0 and y1 > y0:
            crop = self.pil_image.crop((x0, y0, x1, y1))
            size = (max(1, round((x1 - x0) * self.scale)), max(1, round((y1 - y0) * self.scale)))
            resample = Image.Resampling.NEAREST if self.scale >= 4 else Image.Resampling.BILINEAR
            self.tk_image = ImageTk.PhotoImage(crop.resize(size, resample))
            px, py = self.img_to_canvas(x0, y0)
            self.canvas.create_image(px, py, anchor=tk.NW, image=self.tk_image)
        for i, ann in enumerate(self._anns()):
            self._draw_annotation(ann, selected=(i == self.selected_idx))
        self._draw_polygon_preview()
        self._update_scrollbars()

    def _draw_annotation(self, ann, selected=False):
        color = self._color_for_label(ann["label"])
        width = 3 if selected else 2
        dash = (6, 2) if selected else ()
        if ann["type"] == "bbox":
            x0, y0, x1, y1 = ann_bounds(ann)
            cx0, cy0 = self.img_to_canvas(x0, y0)
            cx1, cy1 = self.img_to_canvas(x1, y1)
            self.canvas.create_rectangle(cx0, cy0, cx1, cy1, outline=color, width=width, dash=dash)
            tx, ty = cx0 + 4, cy0 + 2
        else:
            pts = ann_points(ann)
            flat = [c for p in pts for c in self.img_to_canvas(*p)]
            self.canvas.create_polygon(flat, outline=color, fill=color, stipple="gray25",
                                       width=width, dash=dash)
            tx, ty = self.img_to_canvas(*pts[0])
            tx, ty = tx + 4, ty - 14
        self.canvas.create_text(tx, ty, anchor=tk.NW, text=ann["label"], fill=color,
                                font=(MONO, 9, "bold"))
        if selected:
            for hx, hy in (self.img_to_canvas(*p) for p in ann_points(ann)):
                self.canvas.create_oval(hx - HANDLE_RADIUS, hy - HANDLE_RADIUS,
                                        hx + HANDLE_RADIUS, hy + HANDLE_RADIUS,
                                        fill=color, outline="white")

    def _draw_polygon_preview(self):
        self.canvas.delete("poly_preview")
        if not self.polygon_points:
            return
        color = self._current_color()
        cpts = [self.img_to_canvas(*p) for p in self.polygon_points]
        for a, b in zip(cpts, cpts[1:]):
            self.canvas.create_line(*a, *b, fill=color, width=2, tags="poly_preview")
        for i, (cx, cy) in enumerate(cpts):
            r = 6 if i == 0 else 4
            self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, fill=color, outline="white",
                                    width=2 if i == 0 else 1, tags="poly_preview")

    # ── Tools & mouse ────────────────────────────────────────────────────────
    def _on_tool_change(self):
        self._cancel_polygon(redraw=False)
        self.drawing = False
        if self.tool.get() != "select":
            self.selected_idx = None
            self._refresh_ann_list()
        hints = {"select": "Select: click to select · drag to move · drag a handle to resize",
                 "bbox": "BBox: drag to draw a box",
                 "polygon": "Polygon: click to add points · click first point / Enter / double-click to finish"}
        self.status_var.set(hints[self.tool.get()])
        self.redraw()

    def _on_mouse_press(self, event):
        if self.pil_image is None:
            return
        self.canvas.focus_set()
        ix, iy = self.canvas_to_img(event.x, event.y)
        tool = self.tool.get()
        if tool == "select":
            self._select_press(event, ix, iy)
        elif tool == "bbox":
            self.drawing = True
            self.start_x, self.start_y = self._clamp_img(ix, iy)
            self.current_rect = self.canvas.create_rectangle(
                event.x, event.y, event.x, event.y,
                outline=self._current_color(), width=2, dash=(4, 2))
        else:
            self._polygon_press(event, ix, iy)

    def _select_press(self, event, ix, iy):
        sel = self._selected_ann()
        if sel is not None:
            h = self._handle_hit(sel, event.x, event.y)
            if h is not None:
                self._begin_edit(("handle", h), ix, iy, sel)
                return
        hit = self._hit_test(ix, iy)
        self.selected_idx = hit
        if hit is not None:
            self._begin_edit(("move", None), ix, iy, self._anns()[hit])
        self._refresh_ann_list()
        self.redraw()

    def _begin_edit(self, mode, ix, iy, ann):
        self.edit_mode = mode
        self.edit_origin = (ix, iy)
        self.edit_orig = copy.deepcopy(ann)
        self._pending_snap = self._snapshot()
        self._edit_moved = False

    def _on_mouse_drag(self, event):
        if self.pil_image is None:
            return
        if self.edit_mode:
            self._edit_drag(event)
        elif self.drawing and self.current_rect:
            x0, y0 = self.img_to_canvas(self.start_x, self.start_y)
            self.canvas.coords(self.current_rect, x0, y0, event.x, event.y)

    def _edit_drag(self, event):
        ann = self._selected_ann()
        if ann is None:
            return
        rx, ry = self.canvas_to_img(event.x, event.y)
        ox, oy = self.edit_origin
        if not self._edit_moved:
            if math.hypot((rx - ox) * self.scale, (ry - oy) * self.scale) < 2:
                return
            self._edit_moved = True
            self._commit(self._pending_snap)
        mode, h = self.edit_mode
        orig = self.edit_orig
        if mode == "move":
            iw, ih = self.pil_image.size
            bx0, by0, bx1, by1 = ann_bounds(orig)
            dx = max(-bx0, min(iw - bx1, rx - ox))
            dy = max(-by0, min(ih - by1, ry - oy))
            if ann["type"] == "bbox":
                ann["x1"], ann["x2"] = orig["x1"] + dx, orig["x2"] + dx
                ann["y1"], ann["y2"] = orig["y1"] + dy, orig["y2"] + dy
            else:
                ann["points"] = [[px + dx, py + dy] for px, py in orig["points"]]
        else:
            ix, iy = self._clamp_img(rx, ry)
            if ann["type"] == "bbox":
                kx = "x1" if h in (0, 3) else "x2"
                ky = "y1" if h in (0, 1) else "y2"
                ann[kx], ann[ky] = ix, iy
            else:
                ann["points"][h] = [ix, iy]
        self.redraw()

    def _on_mouse_release(self, event):
        if self.edit_mode:
            ann = self._selected_ann()
            if ann is not None and ann["type"] == "bbox":
                x0, y0, x1, y1 = ann_bounds(ann)
                ann.update(x1=x0, y1=y0, x2=x1, y2=y1)
            moved = self._edit_moved
            self.edit_mode = None
            self.edit_orig = None
            self._pending_snap = None
            if moved:
                self._after_change()
            return
        if not self.drawing:
            return
        self.drawing = False
        if self.current_rect:
            self.canvas.delete(self.current_rect)
            self.current_rect = None
        ix, iy = self._clamp_img(*self.canvas_to_img(event.x, event.y))
        if abs(ix - self.start_x) < MIN_BOX_SIZE or abs(iy - self.start_y) < MIN_BOX_SIZE:
            self.redraw()
            return
        label = self._ask_class()
        if not label:
            self.redraw()
            return
        self._push_undo()
        anns = self._anns()
        anns.append(make_bbox(self.start_x, self.start_y, ix, iy, label))
        self.selected_idx = len(anns) - 1
        self.last_label = label
        self._after_change()
        self.status_var.set(f"Added bbox: {label}")

    def _on_mouse_motion(self, event):
        if self.pil_image is None:
            return
        ix, iy = self.canvas_to_img(event.x, event.y)
        iw, ih = self.pil_image.size
        inside = 0 <= ix < iw and 0 <= iy < ih
        pos = f"x:{int(ix)} y:{int(iy)}  " if inside else ""
        self.coord_var.set(f"{pos}zoom {self.scale * 100:.0f}%")
        if self.tool.get() == "polygon" and self.polygon_points:
            self._draw_rubber_band(event)

    def _near_first_point(self, event):
        if len(self.polygon_points) < 3:
            return False
        fx, fy = self.img_to_canvas(*self.polygon_points[0])
        return math.hypot(event.x - fx, event.y - fy) <= SNAP_RADIUS

    def _draw_rubber_band(self, event):
        self.canvas.delete("rubber")
        color = self._current_color()
        px, py = self.img_to_canvas(*self.polygon_points[-1])
        if self._near_first_point(event):
            fx, fy = self.img_to_canvas(*self.polygon_points[0])
            self.canvas.create_line(px, py, fx, fy, fill=color, width=2, tags="rubber")
            self.canvas.create_oval(fx - 9, fy - 9, fx + 9, fy + 9, outline="white", width=2,
                                    dash=(3, 2), tags="rubber")
        else:
            self.canvas.create_line(px, py, event.x, event.y, fill=color, width=2,
                                    dash=(4, 3), tags="rubber")

    # ── Polygon drawing ──────────────────────────────────────────────────────
    def _polygon_press(self, event, ix, iy):
        if self._near_first_point(event):
            self._finish_polygon()
            return
        self.polygon_points.append(self._clamp_img(ix, iy))
        self._draw_polygon_preview()
        self.canvas.delete("rubber")
        self._polygon_status()

    def _polygon_status(self):
        n = len(self.polygon_points)
        hint = ("click first point / Enter to close" if n >= 3 else "need ≥ 3 points")
        self.status_var.set(f"Polygon: {n} pts — {hint} · Backspace: undo point · Esc: cancel")

    def _polygon_undo_point(self):
        if self.polygon_points:
            self.polygon_points.pop()
            self.canvas.delete("rubber")
            self._draw_polygon_preview()
            self._polygon_status()

    def _on_double_click(self, _event):
        if self.tool.get() == "polygon":
            self._finish_polygon()

    def _finish_polygon(self):
        if self.tool.get() != "polygon" or len(self.polygon_points) < 3:
            return
        label = self._ask_class()
        if not label:                    # cancelled → keep polygon in progress
            self._polygon_status()
            return
        pts = list(self.polygon_points)
        self._push_undo()
        self._clear_polygon_preview()
        anns = self._anns()
        anns.append(make_polygon(pts, label))
        self.selected_idx = len(anns) - 1
        self.last_label = label
        self._after_change()
        self.status_var.set(f"Added polygon: {label} ({len(pts)} pts)")

    def _cancel_polygon(self, redraw=True):
        had = bool(self.polygon_points)
        self._clear_polygon_preview()
        if had and redraw:
            self.status_var.set("Polygon cancelled")
            self.redraw()

    def _clear_polygon_preview(self):
        self.polygon_points = []
        self.canvas.delete("poly_preview")
        self.canvas.delete("rubber")

    # ── Hit testing ──────────────────────────────────────────────────────────
    def _hit_test(self, ix, iy):
        """Return index of the smallest annotation under the point (so nested shapes stay pickable)."""
        best, best_area = None, float("inf")
        for i, ann in enumerate(self._anns()):
            if ann_contains(ann, ix, iy):
                area = ann_area(ann)
                if area <= best_area:
                    best, best_area = i, area
        return best

    def _handle_hit(self, ann, cx, cy):
        for i, p in enumerate(ann_points(ann)):
            hx, hy = self.img_to_canvas(*p)
            if math.hypot(cx - hx, cy - hy) <= HANDLE_HIT:
                return i
        return None

    # ── Annotation management ────────────────────────────────────────────────
    def delete_annotation(self):
        if self.selected_idx is None or self._selected_ann() is None:
            return
        self._push_undo()
        del self._anns()[self.selected_idx]
        self.selected_idx = None
        self._after_change()

    def relabel_annotation(self):
        ann = self._selected_ann()
        if ann is None:
            return
        label = ClassDialog(self, self.classes, self._color_for_label,
                            default=ann["label"], title="Relabel").result
        if label and label != ann["label"]:
            self._push_undo()
            ann["label"] = label
            self.last_label = label
            self._after_change()

    def _refresh_ann_list(self):
        self.ann_listbox.delete(0, tk.END)
        for i, ann in enumerate(self._anns()):
            icon = "⬜" if ann["type"] == "bbox" else "⬡"
            self.ann_listbox.insert(tk.END, f"{icon} {ann['label']}")
            self.ann_listbox.itemconfig(i, fg=self._color_for_label(ann["label"]))
        if self.selected_idx is not None:
            self.ann_listbox.selection_set(self.selected_idx)
            self.ann_listbox.see(self.selected_idx)

    def _on_ann_select(self, _event):
        sel = self.ann_listbox.curselection()
        if sel:
            self.selected_idx = sel[0]
            self.redraw()

    # ── Classes ──────────────────────────────────────────────────────────────
    def _rebuild_classes(self):
        """Keep stable order of known classes, drop unused ones, append new ones."""
        used = []
        for anns in self.annotations.values():
            for ann in anns:
                if ann["label"] not in used:
                    used.append(ann["label"])
        kept = [c for c in self.classes if c in used]
        self.classes = kept + [c for c in used if c not in kept]
        self._refresh_classes()

    def _refresh_classes(self):
        self.class_listbox.delete(0, tk.END)
        counts = {}
        for anns in self.annotations.values():
            for ann in anns:
                counts[ann["label"]] = counts.get(ann["label"], 0) + 1
        for i, c in enumerate(self.classes):
            self.class_listbox.insert(tk.END, f"  {i}: {c}  ({counts.get(c, 0)})")
            self.class_listbox.itemconfig(i, fg=COLORS[i % len(COLORS)])
        n = len(self.classes)
        self.cls_count_label.config(text=f"{n} class{'es' if n != 1 else ''}")

    def _rename_class(self, _event=None):
        sel = self.class_listbox.curselection()
        if not sel:
            return
        old = self.classes[sel[0]]
        new = simpledialog.askstring("Rename class",
                                     f"Rename '{old}' in ALL images (existing name merges):",
                                     initialvalue=old, parent=self)
        new = (new or "").strip()
        if not new or new == old:
            return
        for anns in self.annotations.values():
            for ann in anns:
                if ann["label"] == old:
                    ann["label"] = new
        if new not in self.classes:
            self.classes[sel[0]] = new
        self.undo_stack.clear()          # global edit: per-image history no longer valid
        self.redo_stack.clear()
        self.last_label = new if self.last_label == old else self.last_label
        self._after_change()
        self._refresh_img_list()

    def _ask_class(self):
        return ClassDialog(self, self.classes, self._color_for_label, default=self.last_label).result

    # ── Export ───────────────────────────────────────────────────────────────
    def _annotated_items(self):
        items = []
        for p in self.image_paths:
            anns = self.annotations.get(p)
            if anns:
                w, h = self._image_size(p)
                items.append((p, anns, w, h))
        return items

    def export(self, fmt):
        items = self._annotated_items()
        if not items:
            messagebox.showwarning("Nothing to export", "No annotations yet.")
            return
        out_dir = filedialog.askdirectory(title="Select output folder")
        if not out_dir:
            return
        self.save_project()
        try:
            if fmt.startswith("yolo"):
                msg = self._export_yolo(items, out_dir, fmt.split("_")[1])
            elif fmt == "coco":
                msg = self._export_coco(items, out_dir)
            else:
                msg = self._export_voc(items, out_dir)
        except OSError as exc:
            messagebox.showerror("Export failed", str(exc))
            return
        messagebox.showinfo("Export complete", msg)
        self.status_var.set(msg.replace("\n", " "))

    def _export_yolo(self, items, out_dir, mode):
        stems = [os.path.splitext(os.path.basename(p))[0] for p, *_ in items]
        if len(set(stems)) != len(stems):
            messagebox.showwarning("Duplicate names",
                                   "Some images share the same file name (different extension). "
                                   "Their YOLO label files will overwrite each other.")
        labels_dir = os.path.join(out_dir, "labels")
        os.makedirs(labels_dir, exist_ok=True)
        with open(os.path.join(out_dir, "classes.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(self.classes) + "\n")
        with open(os.path.join(out_dir, "data.yaml"), "w", encoding="utf-8") as f:
            f.write(f"nc: {len(self.classes)}\nnames:\n")
            for i, c in enumerate(self.classes):
                f.write(f"  {i}: {json.dumps(c, ensure_ascii=False)}\n")
        for (path, anns, w, h), stem in zip(items, stems):
            lines = []
            for ann in anns:
                cid = self.classes.index(ann["label"])
                if mode == "detect":
                    x0, y0, x1, y1 = ann_bounds(ann)
                    lines.append(f"{cid} {clamp01((x0 + x1) / 2 / w):.6f} {clamp01((y0 + y1) / 2 / h):.6f} "
                                 f"{clamp01((x1 - x0) / w):.6f} {clamp01((y1 - y0) / h):.6f}")
                else:
                    coords = " ".join(f"{clamp01(x / w):.6f} {clamp01(y / h):.6f}"
                                      for x, y in ann_points(ann))
                    lines.append(f"{cid} {coords}")
            with open(os.path.join(labels_dir, stem + ".txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
        kind = "boxes" if mode == "detect" else "polygons"
        return (f"YOLO ({kind}) export: {len(items)} label files in 'labels/', "
                f"plus classes.txt and data.yaml")

    def _export_coco(self, items, out_dir):
        coco = {"info": {"description": f"{APP_NAME} export"}, "licenses": [],
                "images": [], "annotations": [],
                "categories": [{"id": i + 1, "name": c} for i, c in enumerate(self.classes)]}
        ann_id = 1
        for img_id, (path, anns, w, h) in enumerate(items, 1):
            coco["images"].append({"id": img_id, "file_name": os.path.basename(path),
                                   "width": w, "height": h})
            for ann in anns:
                x0, y0, x1, y1 = ann_bounds(ann)
                entry = {"id": ann_id, "image_id": img_id,
                         "category_id": self.classes.index(ann["label"]) + 1,
                         "bbox": [round(x0, 2), round(y0, 2), round(x1 - x0, 2), round(y1 - y0, 2)],
                         "area": round(ann_area(ann), 2), "iscrowd": 0, "segmentation": []}
                if ann["type"] == "polygon":
                    entry["segmentation"] = [[round(c, 2) for p in ann["points"] for c in p]]
                coco["annotations"].append(entry)
                ann_id += 1
        out_file = os.path.join(out_dir, "annotations.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(coco, f, indent=2)
        return f"COCO export: {len(items)} images, {ann_id - 1} annotations\n→ {out_file}"

    def _export_voc(self, items, out_dir):
        for path, anns, w, h in items:
            root = ET.Element("annotation")
            ET.SubElement(root, "folder").text = os.path.basename(os.path.dirname(path))
            ET.SubElement(root, "filename").text = os.path.basename(path)
            ET.SubElement(root, "path").text = path
            size = ET.SubElement(root, "size")
            ET.SubElement(size, "width").text = str(w)
            ET.SubElement(size, "height").text = str(h)
            ET.SubElement(size, "depth").text = "3"
            ET.SubElement(root, "segmented").text = "1" if any(a["type"] == "polygon" for a in anns) else "0"
            for ann in anns:
                obj = ET.SubElement(root, "object")
                ET.SubElement(obj, "name").text = ann["label"]
                ET.SubElement(obj, "pose").text = "Unspecified"
                ET.SubElement(obj, "truncated").text = "0"
                ET.SubElement(obj, "difficult").text = "0"
                x0, y0, x1, y1 = ann_bounds(ann)
                box = ET.SubElement(obj, "bndbox")       # polygons also get their enclosing box
                ET.SubElement(box, "xmin").text = str(round(x0))
                ET.SubElement(box, "ymin").text = str(round(y0))
                ET.SubElement(box, "xmax").text = str(round(x1))
                ET.SubElement(box, "ymax").text = str(round(y1))
                if ann["type"] == "polygon":              # non-standard extension
                    poly = ET.SubElement(obj, "polygon")
                    for i, (x, y) in enumerate(ann["points"], 1):
                        ET.SubElement(poly, f"x{i}").text = str(round(x))
                        ET.SubElement(poly, f"y{i}").text = str(round(y))
            stem = os.path.splitext(os.path.basename(path))[0]
            tree = ET.ElementTree(root)
            ET.indent(tree, space="  ")
            tree.write(os.path.join(out_dir, stem + ".xml"), encoding="utf-8", xml_declaration=True)
        return f"Pascal VOC export: {len(items)} XML files"


# ── Entry point ──────────────────────────────────────────────────────────────
def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else None
    AnnotatorApp(folder).mainloop()


if __name__ == "__main__":
    main()
