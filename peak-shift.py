"""
XRD Peak Shift Comparison
=========================
Overlays smoothed XRD patterns from multiple samples, auto-detects
the dominant peak in each, and produces a publication-quality figure:

  • Full-range overlay panel  — all patterns stacked/offset for clarity
  • Zoomed inset panel        — tight window around the shifting peak,
                                with annotated 2θ positions and Δ2θ shift
                                relative to a reference sample

Dependencies: numpy, pandas, scipy, matplotlib
Python 3.10+  |  No extra packages required.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from scipy.signal  import savgol_filter, find_peaks
from scipy.optimize import curve_fit
from scipy.special  import voigt_profile
import warnings
warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════
#  USER SETTINGS  ← edit these
# ═══════════════════════════════════════════════════════════════════

CSV_FILES    = ["Book1.csv", "Book2.csv", "Book3.csv", "Book0.csv"]
SAMPLE_NAMES = ["3FCZ-TiO2",  "2FCZ-TiO₂", "1FCZ-TiO2",  "Pure TiO₂"]
REFERENCE_IDX = 3          # index of the reference sample for Δ2θ shift

# Smoothing
AA_WINDOW = 20             # Adjacent-averaging window (matches Origin)
SG_WINDOW = 15             # Savitzky-Golay window for peak fitting
SG_POLY   = 3

# Peak detection
PEAK_HEIGHT_FRAC    = 0.05
PEAK_PROMINENCE_FRAC= 0.05
PEAK_DISTANCE       = 20

# Zoom window: ± this many degrees around the spread of detected peaks
ZOOM_MARGIN_DEG = 1.2

# Vertical stacking offset for full-range panel (fraction of max intensity)
# Set to 0 to overlay without offset
STACK_OFFSET_FRAC = 0.15

# ── Publication color palette (Wong 2011 colorblind-safe) ──────────
# One distinct color per sample — works in both color and greyscale print
COLORS = ["#004488",   # deep navy
          "#BB5566",   # muted rose
          "#DDAA33",   # amber
          "#000000"]   # black  ← reference last (most prominent)

LINESTYLES = ["-", "-", "-", "-"]   # all solid; dashes added for zoom panel


# ═══════════════════════════════════════════════════════════════════
#  GLOBAL rcPARAMS  (same journal style as the XRD pipeline)
# ═══════════════════════════════════════════════════════════════════
plt.rcParams.update({
    "font.family":          "serif",
    "font.serif":           ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset":     "stix",
    "font.size":            10,
    "axes.labelsize":       11,
    "xtick.labelsize":      9,
    "ytick.labelsize":      9,
    "legend.fontsize":      9,
    "axes.spines.top":      True,
    "axes.spines.right":    True,
    "xtick.direction":      "in",
    "ytick.direction":      "in",
    "xtick.top":            True,
    "ytick.right":          True,
    "xtick.minor.visible":  True,
    "ytick.minor.visible":  True,
    "xtick.major.size":     5,
    "xtick.minor.size":     2.5,
    "ytick.major.size":     5,
    "ytick.minor.size":     2.5,
    "xtick.major.width":    0.8,
    "ytick.major.width":    0.8,
    "axes.linewidth":       0.8,
    "lines.linewidth":      1.0,
    "axes.grid":            False,
    "legend.frameon":       True,
    "legend.framealpha":    0.92,
    "legend.edgecolor":     "0.75",
    "figure.facecolor":     "white",
    "axes.facecolor":       "white",
})


# ═══════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════
def adjacent_average(y, n):
    """Uniform moving average (Origin 'Adjacent Averaging')."""
    kernel = np.ones(n) / n
    padded = np.pad(y, (n // 2, n // 2), mode="reflect")
    return np.convolve(padded, kernel, mode="valid")[:len(y)]

def voigt_bg(x, amp, center, sigma, gamma, bg0, bg1):
    return amp * voigt_profile(x - center, sigma, gamma) + bg0 + bg1*(x - center)

def voigt_fwhm(sigma, gamma):
    fG = 2*np.sqrt(2*np.log(2))*abs(sigma)
    fL = 2*abs(gamma)
    return 0.5346*fL + np.sqrt(0.2166*fL**2 + fG**2)

def fit_dominant_peak(x, y_smooth, peak_idx, window=60):
    """Fit a Voigt to the dominant peak; return fitted centre and FWHM."""
    lo = max(0, peak_idx - window)
    hi = min(len(x) - 1, peak_idx + window)
    xw = x[lo:hi]; yw = y_smooth[lo:hi]
    amp0 = yw.max() - yw.min()
    try:
        popt, _ = curve_fit(
            voigt_bg, xw, yw,
            p0=[amp0, x[peak_idx], 0.1, 0.1, yw[0], 0.0],
            bounds=([0, xw.min(), 1e-5, 1e-5, -np.inf, -np.inf],
                    [np.inf, xw.max(), 5, 5, np.inf, np.inf]),
            maxfev=10_000)
        _, center, sigma, gamma, _, _ = popt
        return center, voigt_fwhm(sigma, gamma), popt, xw
    except Exception:
        return x[peak_idx], None, None, None


# ═══════════════════════════════════════════════════════════════════
#  LOAD, SMOOTH, DETECT & FIT
# ═══════════════════════════════════════════════════════════════════
records = []   # one dict per sample

for csv_file, name in zip(CSV_FILES, SAMPLE_NAMES):
    df = pd.read_csv(csv_file, header=0)
    df.columns = ["two_theta", "intensity"]
    df = df.dropna().astype(float).sort_values("two_theta").reset_index(drop=True)

    x = df["two_theta"].values
    y = df["intensity"].values

    # Two-stage smoothing: adjacent average for display, S-G for fitting
    y_aa  = adjacent_average(y,  AA_WINDOW)
    y_fit = savgol_filter(y_aa, SG_WINDOW, SG_POLY)

    # Peak detection on the S-G smoothed signal
    i_range = y_fit.max() - y_fit.min()
    peaks, props = find_peaks(
        y_fit,
        height     = y_fit.min() + PEAK_HEIGHT_FRAC * i_range,
        prominence = PEAK_PROMINENCE_FRAC * i_range,
        distance   = PEAK_DISTANCE,
    )
    if len(peaks) == 0:
        raise RuntimeError(f"No peaks found in {csv_file}")

    # Dominant peak = highest prominence
    dom_idx = peaks[np.argmax(props["prominences"])]

    # Voigt fit for precise centre
    center, fwhm, popt, xwin = fit_dominant_peak(x, y_fit, dom_idx)

    records.append({
        "name":    name,
        "x":       x,
        "y_raw":   y,
        "y_aa":    y_aa,       # adjacent-averaged (plotted)
        "y_fit":   y_fit,      # S-G on top (used for fitting only)
        "dom_idx": dom_idx,
        "center":  center,
        "fwhm":    fwhm,
        "popt":    popt,
        "xwin":    xwin,
    })
    print(f"  {name:15s}  dominant peak at 2θ = {center:.4f}°"
          + (f"  FWHM = {fwhm:.4f}°" if fwhm else "  (fit failed)"))

ref_center = records[REFERENCE_IDX]["center"]
print(f"\n  Reference: {SAMPLE_NAMES[REFERENCE_IDX]}  (2θ = {ref_center:.4f}°)")
for r in records:
    shift = r["center"] - ref_center
    r["shift"] = shift
    print(f"  {r['name']:15s}  Δ2θ = {shift:+.4f}°")


# ═══════════════════════════════════════════════════════════════════
#  FIGURE — two-panel layout
#  Left (wider): full-range stacked overlay
#  Right (narrower): zoomed peak window
#  Total width: 7.2 in (full journal page)
# ═══════════════════════════════════════════════════════════════════
fig, (ax_full, ax_zoom) = plt.subplots(
    1, 2,
    figsize=(7.2, 3.2),
    gridspec_kw={"width_ratios": [2.8, 1.8]},
)
fig.subplots_adjust(wspace=0.38)

n = len(records)

# ── Global intensity scale for stacking ──────────────────────────
global_max = max(r["y_aa"].max() for r in records)
offset_step = STACK_OFFSET_FRAC * global_max   # per-sample vertical lift

# ── Panel (a): Full-range stacked overlay ─────────────────────────
for i, r in enumerate(records):
    offset = i * offset_step
    ax_full.plot(r["x"], r["y_aa"] + offset,
                 color=COLORS[i], lw=0.9, ls=LINESTYLES[i],
                 label=r["name"], zorder=n - i)

    # Vertical marker at dominant peak
    yt = r["y_aa"][r["dom_idx"]] + offset
    ax_full.plot(r["center"], yt,
                 marker="|", ms=7, mew=1.2,
                 color=COLORS[i], ls="none", zorder=n+1)

ax_full.set_xlabel(r"2$\theta$ (degrees)")
ax_full.set_ylabel("Intensity (arb. units)")
ax_full.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax_full.xaxis.set_minor_locator(ticker.MultipleLocator(5))
ax_full.yaxis.set_major_locator(ticker.NullLocator())   # stacked → no y numbers
ax_full.set_xlim(records[0]["x"][0] - 0.5, records[0]["x"][-1] + 0.5)
y_top = global_max + (n + 0.3) * offset_step
ax_full.set_ylim(-0.02 * global_max, y_top)
ax_full.legend(loc="upper right", handlelength=1.6,
               borderpad=0.4, labelspacing=0.25, fontsize=8.5)
ax_full.text(0.03, 0.97, "(a)", transform=ax_full.transAxes,
             fontsize=9, fontweight="bold", va="top", fontfamily="serif")

# ── Panel (b): Zoomed peak window ─────────────────────────────────
all_centers = [r["center"] for r in records]
zoom_lo = min(all_centers) - ZOOM_MARGIN_DEG
zoom_hi = max(all_centers) + ZOOM_MARGIN_DEG

# Normalise each pattern to its local max in the zoom window,
# then stack — makes peak shapes directly comparable
for i, r in enumerate(records):
    mask   = (r["x"] >= zoom_lo - 1) & (r["x"] <= zoom_hi + 1)
    x_z    = r["x"][mask]
    y_z    = r["y_aa"][mask]
    y_norm = y_z / y_z.max()            # normalise 0–1 in zoom window
    offset = i * 0.28                   # fixed fractional offset

    ax_zoom.plot(x_z, y_norm + offset,
                 color=COLORS[i], lw=1.0, ls=LINESTYLES[i], zorder=n - i)

    # Vertical dashed line at fitted centre
    peak_norm_y = 1.0 + offset          # normalised peak is at 1
    ax_zoom.plot([r["center"], r["center"]],
                 [offset, peak_norm_y + 0.04],
                 color=COLORS[i], lw=0.7, ls="--", zorder=n+1)

    # 2θ label — above the dashed line, alternating left/right to avoid clash
    ha_side = "left" if i % 2 == 0 else "right"
    dx_pt   = 3 if i % 2 == 0 else -3
    ax_zoom.annotate(
        f"{r['center']:.3f}°",
        xy     = (r["center"], peak_norm_y + 0.06),
        xytext = (dx_pt, 2),
        textcoords = "offset points",
        fontsize   = 7.5,
        fontfamily = "serif",
        color      = COLORS[i],
        ha         = ha_side,
        va         = "bottom",
        fontweight = "semibold",
    )

    # Δ2θ shift label (skip reference sample)
    if i != REFERENCE_IDX and r["shift"] != 0:
        sign = "+" if r["shift"] > 0 else ""
        ax_zoom.annotate(
            f"$\\Delta$2$\\theta$ = {sign}{r['shift']:.3f}°",
            xy     = (r["center"], offset + 0.5),
            xytext = (4, 0),
            textcoords = "offset points",
            fontsize   = 6.5,
            fontfamily = "serif",
            color      = COLORS[i],
            va         = "center",
            ha         = "left",
        )

ax_zoom.set_xlabel(r"2$\theta$ (degrees)")
ax_zoom.set_ylabel("Norm. intensity (arb. units)")
ax_zoom.set_xlim(zoom_lo, zoom_hi)
ax_zoom.yaxis.set_major_locator(ticker.NullLocator())   # normalised → no y numbers
ax_zoom.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
ax_zoom.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
ax_zoom.text(0.04, 0.97, "(b)", transform=ax_zoom.transAxes,
             fontsize=9, fontweight="bold", va="top", fontfamily="serif")

# Reference label inside zoom panel
ax_zoom.text(0.97, 0.03,
             f"Ref: {SAMPLE_NAMES[REFERENCE_IDX]}",
             transform=ax_zoom.transAxes,
             fontsize=6.5, ha="right", va="bottom",
             fontfamily="serif", color=COLORS[REFERENCE_IDX],
             style="italic")

# ── Save ──────────────────────────────────────────────────────────
fig.savefig("fig_peak_shift.png", dpi=300, bbox_inches="tight")
fig.savefig("fig_peak_shift.pdf",           bbox_inches="tight")
print("\nSaved fig_peak_shift.png / .pdf")

# ── Peak shift summary table ──────────────────────────────────────
print("\n" + "=" * 60)
print("PEAK POSITION SUMMARY")
print("=" * 60)
rows = []
for r in records:
    rows.append({
        "Sample":             r["name"],
        "2θ peak (°)":        round(r["center"], 4),
        "FWHM (°)":           round(r["fwhm"], 4) if r["fwhm"] else "—",
        "Δ2θ from ref (°)":   f"{r['shift']:+.4f}",
    })
tbl = pd.DataFrame(rows)
print(tbl.to_string(index=False))
print("=" * 60)
tbl.to_csv("peak_shift_table.csv", index=False)
print("Table saved → peak_shift_table.csv")
