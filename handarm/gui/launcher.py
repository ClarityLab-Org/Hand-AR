"""
Opening GUI Module for Hand-ArM2 Project.

Provides a futuristic Tkinter interface for browsing, downloading,
and launching 3D model interaction modes.
"""

import math
import os
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import messagebox

from handarm import config
from handarm.scanning.sketchfab_client import unified_search, download_glb

# ---------------------------------------------------------------------------
# UI Color Palette
# ---------------------------------------------------------------------------
BG = "#080c10"
BG2 = "#0d1520"
PANEL = "#0f1923"
PANEL2 = "#141e2b"
ACCENT = "#00d4ff"
ACCENT2 = "#0077ff"
ACCENT_DIM = "#003d5c"
GREEN = "#00ff88"
ORANGE = "#ff6b35"
TEXT = "#e8f4fd"
TEXT_DIM = "#4a7a99"
TEXT_MUTE = "#243447"
BORDER = "#1a2d3d"
BORDER_HI = "#00d4ff"

# ---------------------------------------------------------------------------
# UI Fonts
# ---------------------------------------------------------------------------
FONT_TITLE = ("Courier New", 26, "bold")
FONT_SUB = ("Courier New", 10, "normal")
FONT_LABEL = ("Courier New", 9, "normal")
FONT_BTN = ("Courier New", 11, "bold")
FONT_BTN_SM = ("Courier New", 9, "bold")
FONT_MONO = ("Courier New", 8, "normal")
FONT_HDR = ("Courier New", 12, "bold")

PROJECT_DIR = str(config.PROJECT_ROOT)
MODELS_DIR = config.MODELS_DIR
LOCAL_MODELS_DIR = MODELS_DIR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _btn_hover(btn: tk.Button, normal_bg: str, hover_bg: str,
               normal_fg: str = TEXT, hover_fg: str = BG) -> None:
    """Binds mouse hover events to smooth button background/foreground color transitions."""
    btn.bind("<Enter>", lambda e: btn.config(bg=hover_bg, fg=hover_fg))
    btn.bind("<Leave>", lambda e: btn.config(bg=normal_bg, fg=normal_fg))


def _make_toplevel(parent: tk.Widget, title: str, w: int, h: int) -> tk.Toplevel:
    """Creates and configures a styled top-level modal window."""
    win = tk.Toplevel(parent)
    win.title(title)
    win.geometry(f"{w}x{h}")
    win.configure(bg=BG)
    win.resizable(False, False)
    win.grab_set()
    tk.Frame(win, bg=ACCENT, height=2).pack(fill="x")
    return win


def _corner_tag(canvas: tk.Canvas, x: int, y: int, size: int = 10,
                clr: str = ACCENT_DIM) -> None:
    """Draws a technical corner bracket overlay on a canvas."""
    canvas.create_line(x, y + size, x, y, x + size, y, fill=clr, width=1)


def _draw_grid(canvas: tk.Canvas, w: int, h: int, step: int = 28,
               clr: str = "#0a1520") -> None:
    """Draws a subtle background grid pattern on a canvas."""
    for x in range(0, w, step):
        canvas.create_line(x, 0, x, h, fill=clr, width=1)
    for y in range(0, h, step):
        canvas.create_line(0, y, w, y, fill=clr, width=1)


def _all_children(widget: tk.Widget) -> list:
    """Recursively collects all child widgets under a parent widget."""
    children = list(widget.winfo_children())
    for child in widget.winfo_children():
        children.extend(_all_children(child))
    return children


# ---------------------------------------------------------------------------
# Animated header canvas
# ---------------------------------------------------------------------------

class HeaderCanvas(tk.Canvas):
    """Custom animated Canvas widget displaying a dynamic sci-fi rotating graphic."""

    SPEED = 0.6

    def __init__(self, parent, **kw):
        super().__init__(parent, **kw)
        self._angle = 0.0
        self._after_id = None
        self.bind("<Destroy>", lambda e: self._stop())
        self._draw()
        self._tick()

    def _stop(self) -> None:
        if self._after_id:
            self.after_cancel(self._after_id)
            self._after_id = None

    def _tick(self) -> None:
        self._angle = (self._angle + self.SPEED) % 360
        self._draw()
        self._after_id = self.after(16, self._tick)

    def _draw(self) -> None:
        self.delete("all")
        w = int(self["width"])
        h = int(self["height"])
        cx, cy = w // 2, h // 2

        _draw_grid(self, w, h, step=24, clr="#090f18")

        r_outer = 54
        a = math.radians(self._angle)

        for i in range(0, 360, 6):
            ia = math.radians(i)
            x0 = cx + r_outer * math.cos(ia)
            y0 = cy + r_outer * math.sin(ia)
            if i % 12 == 0:
                self.create_oval(x0 - 1.5, y0 - 1.5, x0 + 1.5, y0 + 1.5,
                                 fill=ACCENT_DIM, outline="")

        for k in range(3):
            ak = a + math.radians(k * 120)
            x2 = cx + r_outer * math.cos(ak)
            y2 = cy + r_outer * math.sin(ak)
            self.create_line(cx, cy, x2, y2, fill=ACCENT_DIM, width=1)
            self.create_oval(x2 - 3, y2 - 3, x2 + 3, y2 + 3,
                             fill=ACCENT, outline="")
            self.create_line(x2 - 5, y2, x2 + 5, y2, fill=ACCENT, width=1)
            self.create_line(x2, y2 - 5, x2, y2 + 5, fill=ACCENT, width=1)

        for k in range(3):
            ak = -a * 1.5 + math.radians(k * 120)
            bk = -a * 1.5 + math.radians((k + 1) * 120)
            x1 = cx + 22 * math.cos(ak)
            y1 = cy + 22 * math.sin(ak)
            x2 = cx + 22 * math.cos(bk)
            y2 = cy + 22 * math.sin(bk)
            self.create_line(x1, y1, x2, y2, fill=ACCENT2, width=1)

        self.create_oval(cx - 5, cy - 5, cx + 5, cy + 5,
                         fill=ACCENT, outline=BG, width=2)
        self.create_oval(cx - 2, cy - 2, cx + 2, cy + 2,
                         fill="white", outline="")

        pad = 8
        size = 14
        for bx, by in [(pad, pad), (w - pad - size, pad),
                        (pad, h - pad - size), (w - pad - size, h - pad - size)]:
            _corner_tag(self, bx, by, size, ACCENT_DIM)

        for y in range(0, h, 4):
            self.create_line(0, y, w, y, fill="#0a0f18", width=1)


# ---------------------------------------------------------------------------
# Launcher function
# ---------------------------------------------------------------------------

def launch_ursina(model_path: str, mode: str) -> None:
    """Launches the Ursina 3D viewer subprocess with selected model and mode."""
    interaction_file = os.path.join(PROJECT_DIR, "trial2.py")
    if not os.path.exists(interaction_file):
        messagebox.showerror("Error", "trial2.py not found")
        return

    subprocess.Popen([sys.executable, interaction_file, model_path, mode],
                     cwd=PROJECT_DIR)


def load_laz_for_gui(parent: tk.Misc, laz_path: str, **kwargs):
    """Load a LAZ file and turn decode failures into an actionable dialog."""
    from handarm.geometry.laz_loader import load_laz_to_dataframe

    try:
        return load_laz_to_dataframe(laz_path, **kwargs)
    except RuntimeError as exc:
        messagebox.showerror(
            "LiDAR file could not be loaded",
            f"{os.path.basename(laz_path)} could not be decoded completely.\n\n"
            f"{exc}\n\nRe-copy or re-export the LiDAR file, then verify it "
            "with scripts/diagnose_laz.py.",
            parent=parent,
        )
        return None


# ---------------------------------------------------------------------------
# Model selector modal
# ---------------------------------------------------------------------------

def open_model_selector(root: tk.Tk) -> None:
    """Opens the model selection modal allowing search, download, and launch."""
    if not os.path.exists(MODELS_DIR):
        messagebox.showerror("Error", "models/ folder not found")
        return

    win = _make_toplevel(root, "Select Model", 460, 520)

    hdr = tk.Frame(win, bg=PANEL, pady=0)
    hdr.pack(fill="x")
    tk.Label(hdr, text="// SELECT MODEL", fg=ACCENT, bg=PANEL,
             font=FONT_HDR, padx=20, pady=14).pack(side="left")
    tk.Label(hdr, text=".glb  .obj", fg=TEXT_DIM, bg=PANEL,
             font=FONT_MONO).pack(side="right", padx=20)
    tk.Frame(win, bg=BORDER, height=1).pack(fill="x")

    search_frame = tk.Frame(win, bg=BG2, padx=16, pady=10)
    search_frame.pack(fill="x")

    tk.Label(search_frame, text="QUERY >", fg=ACCENT, bg=BG2,
             font=FONT_MONO).pack(side="left", padx=(0, 8))

    search_var = tk.StringVar(value="car")
    search_entry = tk.Entry(
        search_frame, textvariable=search_var,
        bg=PANEL, fg=TEXT, insertbackground=ACCENT,
        font=("Courier New", 10), relief="flat", bd=0,
        highlightthickness=1, highlightcolor=ACCENT,
        highlightbackground=BORDER,
    )
    search_entry.pack(side="left", fill="x", expand=True, ipady=5, padx=(0, 10))
    search_entry.focus_set()

    search_btn = tk.Button(
        search_frame, text="[ SEARCH ]",
        bg=ACCENT, fg=BG, activebackground=ACCENT2, activeforeground=TEXT,
        font=FONT_BTN_SM, relief="flat", bd=0, cursor="hand2", padx=8,
    )
    search_btn.pack(side="left")
    _btn_hover(search_btn, ACCENT, ACCENT2, BG, TEXT)

    tk.Frame(win, bg=BORDER, height=1).pack(fill="x")

    status = tk.Label(win, text="Enter a query and press SEARCH ...", fg=TEXT_DIM,
                      bg=BG2, font=FONT_MONO, anchor="w", padx=16, pady=6)
    status.pack(fill="x")

    outer = tk.Frame(win, bg=BG, padx=16, pady=12)
    outer.pack(fill="both", expand=True)

    canvas_list = tk.Canvas(outer, bg=BG, highlightthickness=0, bd=0)
    sb = tk.Scrollbar(outer, orient="vertical", command=canvas_list.yview,
                      bg=BG, troughcolor=BG2, width=8)
    inner = tk.Frame(canvas_list, bg=BG)

    inner.bind("<Configure>",
               lambda e: canvas_list.configure(scrollregion=canvas_list.bbox("all")))
    canvas_list.create_window((0, 0), window=inner, anchor="nw")
    canvas_list.configure(yscrollcommand=sb.set)
    canvas_list.pack(side="left", fill="both", expand=True)
    sb.pack(side="right", fill="y")

    canvas_list.bind_all(
        "<MouseWheel>",
        lambda e: canvas_list.yview_scroll(-1 * (e.delta // 120), "units"),
    )

    def _make_hover(r, ab, ar):
        def on_enter(e):
            r.config(bg=PANEL2)
            ab.config(bg=ACCENT)
            ar.config(fg=ACCENT, bg=PANEL2)
            for child in r.winfo_children():
                if child not in (ab, ar):
                    try:
                        child.config(bg=PANEL2)
                        for c2 in child.winfo_children():
                            c2.config(bg=PANEL2)
                    except Exception:
                        pass

        def on_leave(e):
            r.config(bg=PANEL)
            ab.config(bg=ACCENT_DIM)
            ar.config(fg=TEXT_DIM, bg=PANEL)
            for child in r.winfo_children():
                if child not in (ab, ar):
                    try:
                        child.config(bg=PANEL)
                        for c2 in child.winfo_children():
                            c2.config(bg=PANEL)
                    except Exception:
                        pass

        r.bind("<Enter>", on_enter)
        r.bind("<Leave>", on_leave)
        for child in r.winfo_children():
            child.bind("<Enter>", on_enter)
            child.bind("<Leave>", on_leave)
            for c2 in child.winfo_children():
                c2.bind("<Enter>", on_enter)
                c2.bind("<Leave>", on_leave)

    def _make_click(m, r):
        def on_click(e):
            search_btn.config(state="disabled")
            status.config(text=f"Downloading  {m['name']} ...", fg=ORANGE)
            win.update_idletasks()

            def _do_download():
                path = download_glb(m, LOCAL_MODELS_DIR)
                # Schedule result handling on main thread (Bug #4 fix)
                win.after(0, lambda: _on_download_complete(path))

            def _on_download_complete(path):
                search_btn.config(state="normal")
                if path:
                    path = os.path.abspath(path)
                    if not os.path.exists(path):
                        messagebox.showerror("Error", f"File not found:\n{path}")
                        return
                    win.destroy()
                    mode = m.get("mode", "inspect")
                    print(f"Launching {mode.upper()} Mode:", path)
                    launch_ursina(path, mode)
                else:
                    status.config(text="Download failed — try another model",
                                  fg=ORANGE)
                    messagebox.showinfo("Download failed",
                                        "Model could not be downloaded")

            # Run download in background thread (Bug #4 fix: non-blocking)
            threading.Thread(target=_do_download, daemon=True).start()

        for widget in _all_children(r):
            widget.bind("<Button-1>", on_click)
        r.bind("<Button-1>", on_click)

    def _populate(models):
        for widget in inner.winfo_children():
            widget.destroy()

        if not models:
            tk.Label(inner, text="No results found — try a different query",
                     fg=ORANGE, bg=BG, font=FONT_LABEL, pady=20).pack()
            status.config(text="0 results", fg=TEXT_DIM)
            return

        local_models = [m for m in models if m["source"] == "local"]
        web_models = [m for m in models if m["source"] == "web"]
        ordered = local_models + web_models

        visible = 0
        for idx, model in enumerate(ordered):
            if model["source"] == "local" and not os.path.exists(
                    model.get("path", "")):
                continue

            name = model["name"]
            ext = (os.path.splitext(name)[1].upper()
                   if model["source"] == "local" else "[WEB]")
            ext_color = GREEN if model["source"] == "local" else ACCENT

            row = tk.Frame(inner, bg=PANEL, pady=0, cursor="hand2")
            row.pack(fill="x", pady=3)

            accent_bar = tk.Frame(row, bg=ACCENT_DIM, width=3)
            accent_bar.pack(side="left", fill="y")

            tk.Label(row, text=f"{idx+1:02d}", fg=TEXT_MUTE, bg=PANEL,
                     font=FONT_MONO, padx=8, pady=10).pack(side="left")

            info = tk.Frame(row, bg=PANEL)
            info.pack(side="left", fill="both", expand=True, padx=4, pady=6)
            tk.Label(info, text=name, fg=TEXT, bg=PANEL,
                     font=("Courier New", 10, "bold"), anchor="w").pack(fill="x")
            tk.Label(info, text=ext, fg=ext_color, bg=PANEL,
                     font=FONT_MONO, anchor="w").pack(fill="x")

            arrow = tk.Label(row, text="  LOAD  >", fg=TEXT_DIM, bg=PANEL,
                             font=FONT_BTN_SM, padx=12)
            arrow.pack(side="right", fill="y")

            _make_hover(row, accent_bar, arrow)
            _make_click(model, row)
            visible += 1

        local_count = sum(1 for m in ordered if m["source"] == "local")
        web_count = sum(1 for m in ordered if m["source"] == "web")
        status.config(
            text=f"{visible} result(s)  —  {local_count} local  /  {web_count} web",
            fg=TEXT_DIM,
        )

    def _do_search():
        query = search_var.get().strip()
        if not query:
            return
        status.config(text=f'Searching for  "{query}" ...', fg=ACCENT)
        search_btn.config(state="disabled")
        win.update_idletasks()

        try:
            results = unified_search(MODELS_DIR, query=query, use_cache=False)
        except Exception as ex:
            status.config(text=f"Search error: {ex}", fg=ORANGE)
            search_btn.config(state="normal")
            return

        search_btn.config(state="normal")
        _populate(results)

    search_btn.config(command=_do_search)
    search_entry.bind("<Return>", lambda e: _do_search())
    win.after(50, _do_search)

    tk.Frame(win, bg=BORDER, height=1).pack(fill="x")
    tk.Label(win, text="Click a model to download & launch the viewer",
             fg=TEXT_DIM, bg=BG, font=FONT_MONO, pady=8).pack()


# ---------------------------------------------------------------------------
# Controls modal
# ---------------------------------------------------------------------------

def show_controls(root: tk.Tk) -> None:
    """Opens modal window presenting hand gesture controls reference cheat sheet."""
    win = _make_toplevel(root, "Gesture Controls", 420, 400)

    tk.Label(win, text="// GESTURE CONTROLS", fg=ACCENT, bg=BG,
             font=FONT_HDR, pady=14).pack()
    tk.Frame(win, bg=BORDER, height=1).pack(fill="x", padx=20)

    card = tk.Frame(win, bg=PANEL, padx=24, pady=16)
    card.pack(padx=20, pady=16, fill="x")

    sections = [
        ("RIGHT HAND", [
            ("Index finger move", "Rotate model"),
            ("Pinch (close)", "Zoom in"),
            ("Pinch (apart)", "Zoom out"),
            ("Peace sign [paused]", "Screenshot"),
        ]),
        ("LEFT HAND", [
            ("Open palm", "Pause all input"),
            ("Wrist move", "Translate model"),
        ]),
    ]

    for section, rows in sections:
        hdr_row = tk.Frame(card, bg=PANEL)
        hdr_row.pack(fill="x", pady=(10, 4))
        tk.Frame(hdr_row, bg=ACCENT, width=3, height=16).pack(side="left",
                                                                padx=(0, 8))
        tk.Label(hdr_row, text=section, fg=ACCENT, bg=PANEL,
                 font=("Courier New", 9, "bold")).pack(side="left")

        for gesture, action in rows:
            row = tk.Frame(card, bg=PANEL)
            row.pack(fill="x", pady=2)
            tk.Label(row, text=f"  {gesture}", fg=TEXT_DIM, bg=PANEL,
                     font=FONT_MONO, width=26, anchor="w").pack(side="left")
            tk.Label(row, text="->", fg=ACCENT_DIM, bg=PANEL,
                     font=FONT_MONO).pack(side="left", padx=4)
            tk.Label(row, text=action, fg=TEXT, bg=PANEL,
                     font=("Courier New", 9, "bold"), anchor="w").pack(side="left")

    tip = tk.Frame(win, bg=ACCENT_DIM, padx=14, pady=10)
    tip.pack(padx=20, fill="x")
    tk.Label(tip, text="TIP  Hold open palm for ~0.25s to engage pause.",
             fg=ACCENT, bg=ACCENT_DIM, font=FONT_MONO, anchor="w").pack(fill="x")

    close_btn = tk.Button(
        win, text="[ CLOSE ]", width=14,
        bg=PANEL, fg=TEXT_DIM,
        activebackground=BORDER, activeforeground=TEXT,
        font=FONT_BTN_SM, relief="flat", bd=0, cursor="hand2",
        command=win.destroy,
    )
    close_btn.pack(pady=16)
    _btn_hover(close_btn, PANEL, ACCENT, TEXT_DIM, BG)


def _count_models() -> int:
    """Counts available 3D model/point cloud files in models/ folder."""
    if os.path.exists(MODELS_DIR):
        return len([f for f in os.listdir(MODELS_DIR)
                    if f.lower().endswith((".glb", ".obj", ".csv"))])
    return 0


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

def main() -> None:
    """Main entry point for the GUI launcher."""
    root = tk.Tk()
    root.title("3D Interaction Controller")
    root.geometry("620x420")
    root.configure(bg=BG)
    root.resizable(False, False)

    tk.Frame(root, bg=ACCENT, height=2).pack(fill="x")

    body = tk.Frame(root, bg=BG)
    body.pack(fill="both", expand=True)

    left = tk.Frame(body, bg=BG2, width=180)
    left.pack(side="left", fill="y")
    left.pack_propagate(False)

    anim = HeaderCanvas(left, width=180, height=300, bg=BG2, highlightthickness=0)
    anim.pack(pady=(30, 0))

    tk.Label(left, text="v2.0.0", fg=TEXT_MUTE, bg=BG2,
             font=FONT_MONO).pack(side="bottom", pady=10)
    tk.Label(left, text="BUILD", fg=TEXT_MUTE, bg=BG2,
             font=FONT_MONO).pack(side="bottom")

    tk.Frame(body, bg=BORDER, width=1).pack(side="left", fill="y")

    right = tk.Frame(body, bg=BG, padx=36)
    right.pack(side="left", fill="both", expand=True)

    tk.Frame(right, bg=BG, height=40).pack()

    tk.Label(right, text="// GESTURE-BASED 3D VIEWER", fg=ACCENT, bg=BG,
             font=FONT_LABEL, anchor="w").pack(fill="x")

    tk.Label(right, text="3D INTERACTION\nCONTROLLER", fg=TEXT, bg=BG,
             font=FONT_TITLE, justify="left", anchor="w", pady=4).pack(fill="x")

    tk.Label(right, text="Real-time hand gesture control for 3D model viewing",
             fg=TEXT_DIM, bg=BG, font=FONT_SUB, wraplength=340, justify="left",
             anchor="w").pack(fill="x", pady=(0, 28))

    tk.Frame(right, bg=BORDER, height=1).pack(fill="x", pady=(0, 24))

    btn_area = tk.Frame(right, bg=BG)
    btn_area.pack(anchor="w")

    start_btn = tk.Button(
        btn_area,
        text="  GET STARTED  >",
        width=18, height=1,
        bg=ACCENT, fg=BG,
        activebackground=ACCENT2, activeforeground=TEXT,
        font=FONT_BTN, relief="flat", bd=0, cursor="hand2",
        pady=10,
        command=lambda: open_model_selector(root),
    )
    start_btn.grid(row=0, column=0, padx=(0, 10), pady=4)
    _btn_hover(start_btn, ACCENT, ACCENT2, BG, TEXT)

    ctrl_btn = tk.Button(
        btn_area,
        text="  VIEW CONTROLS",
        width=18, height=1,
        bg=PANEL, fg=TEXT_DIM,
        activebackground=PANEL2, activeforeground=TEXT,
        font=FONT_BTN, relief="flat", bd=0, cursor="hand2",
        pady=10,
        command=lambda: show_controls(root),
    )
    ctrl_btn.grid(row=0, column=1, pady=4)
    _btn_hover(ctrl_btn, PANEL, PANEL2, TEXT_DIM, TEXT)

    tk.Frame(right, bg=BG, height=20).pack()
    status_row = tk.Frame(right, bg=BG)
    status_row.pack(anchor="w")

    n_models = _count_models()
    dot_color = GREEN if n_models > 0 else ORANGE
    tk.Label(status_row, text="●", fg=dot_color, bg=BG,
             font=("Courier New", 10)).pack(side="left")
    tk.Label(status_row, text=f"  {n_models} model(s) in models/",
             fg=TEXT_DIM, bg=BG, font=FONT_MONO).pack(side="left")

    tk.Frame(root, bg=BORDER, height=1).pack(fill="x")
    foot = tk.Frame(root, bg=BG2, pady=6)
    foot.pack(fill="x")
    tk.Label(foot, text="Place .glb or .obj files in the models/ folder",
             fg=TEXT_MUTE, bg=BG2, font=FONT_MONO).pack(side="left", padx=16)
    tk.Label(foot, text="[  READY  ]", fg=GREEN, bg=BG2,
             font=FONT_MONO).pack(side="right", padx=16)

    root.mainloop()


if __name__ == "__main__":
    main()
