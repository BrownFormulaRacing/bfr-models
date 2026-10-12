# manual mf5.2 lateral tuner: sliders for all 18 coeffs over the measured data.
# defaults are the fitted coeffs from tire.py, save writes a separate _tuned file.
# run from repo root: python -m src.models.tire_tuner [--scale 1.25]

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox, RadioButtons, CheckButtons, Button
from matplotlib.transforms import Bbox

from .tire import Tire, LAT_MF52_COEFF_NAMES, LAT_MF52_LB, LAT_MF52_UB, SA_RANGE, IA_BUCKETS, FZ_TOL, IA_TOL, P_TOL
from ..utils.files import DEFAULT_LAT_MF52_TUNED_COEFFS_PATH

TEST_FZ = [220, 440, 665, 890, 1110] # nominal test loads [N]
START_P = 82.7371 # 12 psi
START_IA = 0
START_FZ = 665
KPA_PER_PSI = 6.89476

FZ_COLORS = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"] # light -> heavy load
IA_COLORS = ["#f2a07a", "#eb6834", "#b4471c"] # 0 -> 4 deg

# slider groups in coeff order
GROUPS = [("C  shape", 1), ("D  peak mu", 3), ("E  curvature", 4),
          ("K  cornering stiffness", 3), ("H  horizontal shift", 3), ("V  vertical shift", 4)]

FIG_SIZE = (16, 9)
BASE_DPI = 80 # 1280x720 window at scale 1, fits a 13" laptop
HEADER_H = 0.022
ROW_H = 0.0435


class LazyTextBox(TextBox):
    # stock stop_typing does a full canvas.draw() per box on every click anywhere
    def stop_typing(self):
        if self.capturekeystrokes:
            super().stop_typing()


class Tuner:
    def __init__(self, tire, scale=1.0):
        self.tire = tire
        self.scale = scale
        self.d = tire.run_data

        self.pressures = sorted(tire.lat_mf52_coeffs)
        self.fz0 = {p: tire.lat_mf52_coeffs[p]["fz0"] for p in self.pressures}
        self.fitted = {p: np.array([tire.lat_mf52_coeffs[p][n] for n in LAT_MF52_COEFF_NAMES]) for p in self.pressures}
        self.tuned = {p: v.copy() for p, v in self.fitted.items()}
        self.fit_rms = {p: self.rms(p, self.fitted[p]) for p in self.pressures}

        self.p = min(self.pressures, key=lambda p: abs(p - START_P))
        self.ia = START_IA
        self.fz = START_FZ
        self.show_fit = False

        self.build()
        self.load_pressure()


    # ---------- model ----------

    def curve(self, params, ia, fz):
        return Tire._lat_mf52_model(params, np.deg2rad(SA_RANGE), np.deg2rad(ia), fz, self.fz0[self.p])


    def rms(self, p, params):
        d = self.d
        at_p = np.abs(d["P"] - p) < P_TOL
        if not at_p.any():
            return float("nan")
        fy = Tire._lat_mf52_model(params, np.deg2rad(d["SA"][at_p]), np.deg2rad(d["IA"][at_p]), d["FZ"][at_p], self.fz0[p])
        return float(np.sqrt(np.mean((fy - d["FY"][at_p]) ** 2)))


    # ---------- layout ----------

    def build(self):
        self.fig = plt.figure(figsize=FIG_SIZE, dpi=BASE_DPI * self.scale)
        if self.fig.canvas.manager is not None:
            self.fig.canvas.manager.set_window_title(f"[{self.tire.run_id}] mf5.2 lateral tuner")

        self.ax_fz = self.fig.add_axes([0.05, 0.56, 0.48, 0.38])
        self.ax_ia = self.fig.add_axes([0.05, 0.08, 0.48, 0.38])
        self.fz_dots, self.fz_fits, self.fz_lines = self.make_series(self.ax_fz, FZ_COLORS, [f"F_Z = {fz} N" for fz in TEST_FZ])
        self.ia_dots, self.ia_fits, self.ia_lines = self.make_series(self.ax_ia, IA_COLORS, [f"IA = {ia}°" for ia in IA_BUCKETS])

        self.build_controls()
        self.build_sliders()

        # full redraws are ~200 ms, so dragging only blits the regions that change.
        # region key -> animated artists in it, later artists draw on top
        self.anim = {"fz": [*self.fz_lines, *self.fz_fits], "ia": [*self.ia_lines, *self.ia_fits], "rms": [self.rms_text],
                     "alpha": [self.s_alpha.poly, self.s_alpha._handle]}
        for i, (sl, tb) in enumerate(zip(self.sliders, self.boxes)):
            tb.text_disp.set_clip_box(tb.ax.bbox)
            self.anim[i] = [sl.poly, sl.vline, sl._handle, tb.text_disp]
        for arts in self.anim.values():
            for a in arts:
                a.set_animated(True)

        self.bgs = None
        self.fig.canvas.mpl_connect("draw_event", self.on_draw)


    def make_series(self, ax, colors, labels):
        dots, fits, lines = [], [], []
        for c, label in zip(colors, labels):
            dots.append(ax.scatter(np.empty(0), np.empty(0), s=6, color=c, alpha=0.35))
            lines.append(ax.plot([], [], color=c, lw=2, label=label)[0])
            # dark dashes so the original fit shows even when tuned == fit
            fits.append(ax.plot([], [], color="0.15", lw=1.2, ls=(0, (4, 3)), alpha=0.9)[0])

        ax.axhline(0, linestyle="--", color="k", alpha=0.25)
        ax.axvline(0, linestyle="--", color="k", alpha=0.25)
        ax.set_xlim(SA_RANGE[0], SA_RANGE[-1])
        ax.set_xlabel("slip angle  α  [deg]")
        ax.set_ylabel("lateral force  F_Y  [N]")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")
        return dots, fits, lines


    def build_controls(self):
        x, w = 0.565, 0.1
        p_labels = [f"{p / KPA_PER_PSI:.0f} psi" for p in self.pressures]
        ia_labels = [f"{ia}°" for ia in IA_BUCKETS]
        fz_labels = [f"{fz} N" for fz in TEST_FZ]

        ax = self.fig.add_axes([x, 0.78, w, 0.13])
        ax.set_title("pressure", fontsize=9)
        self.r_p = RadioButtons(ax, p_labels, active=self.pressures.index(self.p))
        self.r_p.on_clicked(lambda label: self.set_p(self.pressures[p_labels.index(label)]))

        ax = self.fig.add_axes([x, 0.58, w, 0.13])
        ax.set_title("camber (top plot)", fontsize=9)
        self.r_ia = RadioButtons(ax, ia_labels, active=IA_BUCKETS.index(self.ia))
        self.r_ia.on_clicked(lambda label: self.set_ia(IA_BUCKETS[ia_labels.index(label)]))

        ax = self.fig.add_axes([x, 0.33, w, 0.18])
        ax.set_title("load (bottom plot)", fontsize=9)
        self.r_fz = RadioButtons(ax, fz_labels, active=TEST_FZ.index(self.fz))
        self.r_fz.on_clicked(lambda label: self.set_fz(TEST_FZ[fz_labels.index(label)]))

        ax = self.fig.add_axes([x, 0.265, w, 0.045])
        self.c_fit = CheckButtons(ax, ["show fit"], [self.show_fit])
        self.c_fit.on_clicked(self.toggle_fit)

        # opacity of the tuned curves, 0 hides them
        self.fig.text(x, 0.243, "curve opacity", fontsize=9)
        self.s_alpha = Slider(self.fig.add_axes([x + 0.005, 0.215, w - 0.01, 0.02]), "", 0, 1,
                              valinit=1, initcolor="none")
        self.s_alpha.valtext.set_visible(False)
        self.s_alpha.drawon = False
        self.s_alpha.on_changed(self.set_line_alpha)

        self.rms_text = self.fig.text(x, 0.195, "", fontsize=9, va="top", family="monospace")
        self.status_text = self.fig.text(x, 0.015, "", fontsize=8, color="0.4")

        self.b_reset = Button(self.fig.add_axes([x, 0.085, w, 0.04]), "reset to fit")
        self.b_reset.on_clicked(self.reset)
        self.b_save = Button(self.fig.add_axes([x, 0.035, w, 0.04]), "save tuned")
        self.b_save.on_clicked(self.save)


    def build_sliders(self):
        # slider range = fit bounds, widened to cover every fitted value
        fitted = np.array(list(self.fitted.values()))
        lo = np.minimum(LAT_MF52_LB, fitted.min(axis=0))
        hi = np.maximum(LAT_MF52_UB, fitted.max(axis=0))

        self.sliders, self.boxes = [], []
        y, i = 0.96, 0
        for header, n in GROUPS:
            y -= HEADER_H
            self.fig.text(0.695, y + 0.004, header, fontsize=8, color="0.4")
            for _ in range(n):
                y -= ROW_H
                v = self.tuned[self.p][i]

                s = Slider(self.fig.add_axes([0.73, y, 0.185, ROW_H * 0.6]), LAT_MF52_COEFF_NAMES[i],
                           lo[i], hi[i], valinit=v, initcolor="0.35")
                s.valtext.set_visible(False)
                s.drawon = False
                s.on_changed(lambda val, i=i: self.on_slide(i, val))

                # hovercolor = color so hovering doesn't force a redraw
                tb = LazyTextBox(self.fig.add_axes([0.925, y, 0.065, ROW_H * 0.6]), "", initial=f"{v:.6g}",
                                 color=".95", hovercolor=".95")
                tb.on_submit(lambda text, i=i: self.on_type(i, text))

                self.sliders.append(s)
                self.boxes.append(tb)
                i += 1


    # ---------- callbacks ----------

    def on_slide(self, i, val):
        self.tuned[self.p][i] = val
        self.set_text(i, val)
        self.update_curves()
        self.refresh(i)


    def on_type(self, i, text):
        # clicking out of an untouched box shouldn't round the value
        if text.strip() == f"{self.tuned[self.p][i]:.6g}":
            return
        try:
            val = float(text)
        except ValueError:
            val = np.nan
        if not np.isfinite(val):
            self.set_text(i, self.tuned[self.p][i])
            return

        # widen the slider if the typed value is out of range
        s = self.sliders[i]
        if not s.valmin <= val <= s.valmax:
            s.valmin, s.valmax = min(s.valmin, val), max(s.valmax, val)
            s.ax.set_xlim(s.valmin, s.valmax)
            s.poly.set_x(s.valmin)
            self.fig.canvas.draw_idle()
        s.set_val(val)


    def set_text(self, i, val):
        # skip TextBox.set_val, it does a full sync redraw
        self.boxes[i].text_disp.set_text(f"{val:.6g}")


    def set_p(self, p):
        self.p = p
        self.load_pressure()


    def set_ia(self, ia):
        self.ia = ia
        self.update_data()
        self.update_curves()
        self.fig.canvas.draw_idle()


    def set_fz(self, fz):
        self.fz = fz
        self.update_data()
        self.update_curves()
        self.fig.canvas.draw_idle()


    def set_line_alpha(self, val):
        for line in self.fz_lines + self.ia_lines:
            line.set_alpha(val)
        self.refresh("alpha")


    def toggle_fit(self, _):
        self.show_fit = not self.show_fit
        self.update_curves()
        self.refresh()


    def reset(self, _):
        self.tuned[self.p] = self.fitted[self.p].copy()
        self.load_pressure()
        self.status_text.set_text(f"reset {self.p / KPA_PER_PSI:.0f} psi to fit")


    def save(self, _):
        # same format as the fitted file so Tire can load it via mf52_coeffs_path
        coeffs = dict()
        for p in self.pressures:
            c = dict(zip(LAT_MF52_COEFF_NAMES, self.tuned[p].astype(float)))
            c["fz0"] = self.fz0[p]
            c["rms"] = self.rms(p, self.tuned[p])
            coeffs[p] = c

        self.tire.lat_mf52_coeffs = coeffs
        self.tire.mf52_coeffs_path = DEFAULT_LAT_MF52_TUNED_COEFFS_PATH
        self.tire.save_lat_mf52_coeffs()
        self.tire.print_lat_mf52_coeffs()

        print(f"\nsaved tuned coeffs to {DEFAULT_LAT_MF52_TUNED_COEFFS_PATH}")
        self.status_text.set_text(f"saved {DEFAULT_LAT_MF52_TUNED_COEFFS_PATH.name}")
        self.fig.canvas.draw_idle()


    # ---------- drawing ----------

    def load_pressure(self):
        vals, fit = self.tuned[self.p], self.fitted[self.p]
        for i, s in enumerate(self.sliders):
            # marker line on each slider = fitted value
            s.valinit = fit[i]
            s.vline.set_xdata([fit[i]])
            s.eventson = False
            s.set_val(vals[i])
            s.eventson = True
            self.set_text(i, vals[i])

        # fixed y range per pressure so curves don't jump while tuning
        at_p = np.abs(self.d["P"] - self.p) < P_TOL
        lim = 1.1 * np.abs(self.d["FY"][at_p]).max() if at_p.any() else 3000
        for ax in (self.ax_fz, self.ax_ia):
            ax.set_ylim(-lim, lim)

        self.update_data()
        self.update_curves()
        self.fig.canvas.draw_idle()


    def update_data(self):
        d = self.d
        at_p = np.abs(d["P"] - self.p) < P_TOL

        for dots, fz in zip(self.fz_dots, TEST_FZ):
            idx = at_p & (np.abs(d["FZ"] - fz) < FZ_TOL) & (np.abs(d["IA"] - self.ia) < IA_TOL)
            dots.set_offsets(np.c_[d["SA"][idx], d["FY"][idx]])

        for dots, ia in zip(self.ia_dots, IA_BUCKETS):
            idx = at_p & (np.abs(d["FZ"] - self.fz) < FZ_TOL) & (np.abs(d["IA"] - ia) < IA_TOL)
            dots.set_offsets(np.c_[d["SA"][idx], d["FY"][idx]])

        psi = self.p / KPA_PER_PSI
        self.ax_fz.set_title(f"Fy vs SA per Fz @ IA = {self.ia}°, P = {psi:.0f} psi")
        self.ax_ia.set_title(f"Fy vs SA per IA @ Fz = {self.fz} N, P = {psi:.0f} psi")


    def update_curves(self):
        vals, fit = self.tuned[self.p], self.fitted[self.p]

        for line, fit_line, fz in zip(self.fz_lines, self.fz_fits, TEST_FZ):
            line.set_data(SA_RANGE, self.curve(vals, self.ia, fz))
            fit_line.set_data(SA_RANGE, self.curve(fit, self.ia, fz))
            fit_line.set_visible(self.show_fit)

        for line, fit_line, ia in zip(self.ia_lines, self.ia_fits, IA_BUCKETS):
            line.set_data(SA_RANGE, self.curve(vals, ia, self.fz))
            fit_line.set_data(SA_RANGE, self.curve(fit, ia, self.fz))
            fit_line.set_visible(self.show_fit)

        self.rms_text.set_text(f"rms @ {self.p / KPA_PER_PSI:.0f} psi\n"
                               f"tuned {self.rms(self.p, vals):6.1f} N\n"
                               f"fit   {self.fit_rms[self.p]:6.1f} N")


    def on_draw(self, _):
        # cache the static background of each region, then draw the animated artists on it
        cv = self.fig.canvas
        pad = 8 * self.fig.dpi / 72 # slider handle overhangs the track ends
        boxes = {"fz": self.ax_fz.bbox, "ia": self.ax_ia.bbox,
                 "rms": self.rms_text.get_window_extent(cv.get_renderer()).padded(pad)}
        for i, (sl, tb) in enumerate(zip(self.sliders, self.boxes)):
            b = sl.ax.bbox
            boxes[i] = Bbox.union([Bbox.from_extents(b.x0 - pad, b.y0, b.x1 + pad, b.y1), tb.ax.bbox])
        b = self.s_alpha.ax.bbox
        boxes["alpha"] = Bbox.from_extents(b.x0 - pad, b.y0, b.x1 + pad, b.y1)

        # +2 px so antialiased edges are inside the cached region
        self.bgs = {k: cv.copy_from_bbox(b.padded(2)) for k, b in boxes.items()}
        for arts in self.anim.values():
            for a in arts:
                self.fig.draw_artist(a)


    def refresh(self, row=None):
        # redraw plots + rms, plus one slider region if given
        cv = self.fig.canvas
        if self.bgs is None or not cv.supports_blit:
            cv.draw_idle()
            return

        for k in ["fz", "ia", "rms"] + ([row] if row is not None else []):
            cv.restore_region(self.bgs[k])
            for a in self.anim[k]:
                self.fig.draw_artist(a)
        cv.blit(self.fig.bbox)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="manual mf5.2 lateral tuner")
    parser.add_argument("--scale", type=float, default=1.0, help="window scale, 1 = 1280x720")
    args = parser.parse_args()
    if args.scale <= 0:
        parser.error("--scale must be positive")

    tire = Tire()
    tire.setup()

    if getattr(tire, "run_data", None) is None or not tire.lat_mf52_coeffs:
        raise SystemExit("couldn't load tire data or mf5.2 coeffs, make sure you run this from the repo root")

    tuner = Tuner(tire, scale=args.scale)
    plt.show()
