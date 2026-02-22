"""
XRD Analysis Pipeline
=====================
Dependencies: numpy, pandas, scipy, matplotlib  (standard scientific Python stack)
Python 3.10+  |  No lmfit required.

Steps
-----
1.  Load CSV  (columns: <2Theta>, <I>)
2.  Smooth intensity with a Savitzky-Golay filter
3.  Detect diffraction peaks automatically
4.  Fit each peak with an exact Voigt profile + linear background (scipy curve_fit)
    → extract centre (2θ), FWHM (β)
5.  Debye-Scherrer crystallite size:   D = Kλ / (β cosθ)
6.  Williamson-Hall analysis:          β cosθ = Kλ/D + 4ε sinθ
    → D (W-H),  micro-strain ε
7.  Dislocation density:               δ = 1/D²  (×10⁻³ nm⁻²)
8.  Print / save results table
9.  Save two plots:  XRD pattern  +  Williamson-Hall
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # non-interactive backend (change to "Qt5Agg" for live display)
import matplotlib.pyplot as plt
from scipy.signal  import savgol_filter, find_peaks
from scipy.optimize import curve_fit
from scipy.special  import voigt_profile   # exact Voigt  (scipy ≥ 1.7)
from scipy.stats    import linregress
import warnings
warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════
#  USER SETTINGS  ← edit these
# ═══════════════════════════════════════════════════════════════════
CSV_FILE  = "Book3.csv"                 # ← put your file path here
K         = 0.9                         # Scherrer constant
LAMBDA_NM = 0.15406                     # Cu Kα wavelength (nm)

# Savitzky-Golay smoothing
SG_WINDOW = 15                          # must be odd; raise for noisier data
SG_POLY   = 3

# Peak detection  (fractions of intensity range)
PEAK_HEIGHT_FRAC = 0.05    # minimum peak height relative to (max-min)
PEAK_PROMINENCE  = 0.03    # minimum prominence relative to (max-min)
PEAK_DISTANCE    = 20      # minimum peak-to-peak separation in data points
PEAK_WINDOW_PTS  = 50      # half-window around each peak used for fitting

# FWHM sanity filter — discard fits with unrealistic FWHM (°)
FWHM_MIN_DEG = 0.05        # narrower than this ≈ instrumental artefact
FWHM_MAX_DEG = 5.0         # wider than this ≈ failed fit


# ═══════════════════════════════════════════════════════════════════
#  1.  LOAD DATA
# ═══════════════════════════════════════════════════════════════════
print("=" * 65)
print("STEP 1  Loading CSV …")
df = pd.read_csv(CSV_FILE, header=0)
df.columns = ["two_theta", "intensity"]
df = (df.dropna()
        .astype(float)
        .sort_values("two_theta")
        .reset_index(drop=True))

two_theta = df["two_theta"].values   # degrees (2θ)
intensity = df["intensity"].values   # arbitrary units

print(f"  {len(df):,} data points  |  "
      f"2θ range: {two_theta[0]:.2f}° – {two_theta[-1]:.2f}°")


# ═══════════════════════════════════════════════════════════════════
#  2.  SMOOTHING
#
#  Two-stage approach matching Origin's "Adjacent Averaging":
#    a) Adjacent averaging (uniform moving average, window = AA_WINDOW)
#       — identical to Origin's "Adjacent Averaging" method.
#       Uses np.convolve with a flat kernel; edges are handled by
#       reflecting the signal so the output has the same length.
#    b) Optional light Savitzky-Golay pass (for peak fitting only)
#       — SG better preserves peak shapes; used internally for fitting.
#       The plotted curve uses the adjacent-averaged signal.
# ═══════════════════════════════════════════════════════════════════
AA_WINDOW = 20          # ← matches Origin's "Adjacent Averaging, points = 20"

print(f"\nSTEP 2  Smoothing (Adjacent Averaging, window = {AA_WINDOW} pts) …")

# Uniform (box) moving average — same as Origin's Adjacent Averaging
kernel   = np.ones(AA_WINDOW) / AA_WINDOW
# 'reflect' padding avoids edge artefacts (equivalent to Origin behaviour)
padded   = np.pad(intensity, (AA_WINDOW // 2, AA_WINDOW // 2), mode="reflect")
smoothed = np.convolve(padded, kernel, mode="valid")[:len(intensity)]

print(f"  Adjacent-averaging window = {AA_WINDOW} pts")

# Light SG pass kept for peak fitting (better shape preservation)
smoothed_fit = savgol_filter(smoothed, window_length=SG_WINDOW, polyorder=SG_POLY)
print(f"  Additional S-G pass for fitting: window = {SG_WINDOW}, order = {SG_POLY}")


# ═══════════════════════════════════════════════════════════════════
#  3.  AUTOMATIC PEAK DETECTION
# ═══════════════════════════════════════════════════════════════════
print("\nSTEP 3  Peak detection …")
i_range = smoothed_fit.max() - smoothed_fit.min()
peak_idx, _ = find_peaks(
    smoothed_fit,
    height     = smoothed_fit.min() + PEAK_HEIGHT_FRAC * i_range,
    prominence = PEAK_PROMINENCE * i_range,
    distance   = PEAK_DISTANCE,
)
print(f"  {len(peak_idx)} candidate peaks found at "
      f"2θ = {np.round(two_theta[peak_idx], 2).tolist()}")


# ═══════════════════════════════════════════════════════════════════
#  4.  VOIGT PROFILE FITTING
#
#  Model:  f(x) = A · V(x − x₀; σ, γ) + c₀ + c₁(x − x₀)
#           V = exact Voigt  (scipy.special.voigt_profile)
#
#  FWHM of Voigt (Olivero & Longbothum, 1977):
#    fwhm ≈ 0.5346 fL + √(0.2166 fL² + fG²)
#    fG = 2√(2 ln2) σ  (Gaussian component)
#    fL = 2 γ          (Lorentzian component)
# ═══════════════════════════════════════════════════════════════════
print("\nSTEP 4  Voigt profile fitting …")

def _voigt_model(x, amp, center, sigma, gamma, bg0, bg1):
    """Voigt profile with linear background."""
    return amp * voigt_profile(x - center, sigma, gamma) + bg0 + bg1 * (x - center)

def _voigt_fwhm(sigma, gamma):
    """Approximate FWHM of an exact Voigt profile."""
    fG = 2.0 * np.sqrt(2.0 * np.log(2.0)) * abs(sigma)
    fL = 2.0 * abs(gamma)
    return 0.5346 * fL + np.sqrt(0.2166 * fL**2 + fG**2)

fit_results = []   # each entry: dict with fit metadata

for idx in peak_idx:
    # ── Extract local window around peak ──────────────────────────
    lo = max(0, idx - PEAK_WINDOW_PTS)
    hi = min(len(two_theta) - 1, idx + PEAK_WINDOW_PTS)
    x_win = two_theta[lo:hi]
    y_win = smoothed_fit[lo:hi]
    if len(x_win) < 8:
        continue

    # ── Initial guesses ──────────────────────────────────────────
    amp_guess = y_win.max() - y_win.min()
    cen_guess = two_theta[idx]

    try:
        popt, _ = curve_fit(
            _voigt_model, x_win, y_win,
            p0     = [amp_guess, cen_guess, 0.1, 0.1, y_win[0], 0.0],
            bounds = ([0, x_win.min(), 1e-5, 1e-5, -np.inf, -np.inf],
                      [np.inf, x_win.max(), 5.0, 5.0, np.inf, np.inf]),
            maxfev = 10_000,
        )
        _, center, sigma, gamma, _, _ = popt
        fwhm_deg = _voigt_fwhm(sigma, gamma)

        # ── Sanity-check FWHM ────────────────────────────────────
        if not (FWHM_MIN_DEG < fwhm_deg < FWHM_MAX_DEG):
            print(f"  [skip] 2θ≈{two_theta[idx]:.2f}°  "
                  f"FWHM={fwhm_deg:.3f}° out of range [{FWHM_MIN_DEG},{FWHM_MAX_DEG}]°")
            continue

        # ── Degrees → radians for Scherrer / W-H ─────────────────
        theta_rad = np.radians(center / 2.0)   # θ = 2θ / 2
        beta_rad  = np.radians(fwhm_deg)        # β in radians

        fit_results.append({
            "two_theta_deg": center,
            "theta_rad":     theta_rad,
            "fwhm_deg":      fwhm_deg,
            "beta_rad":      beta_rad,
            "popt":          popt,
            "x_win":         x_win,
        })
        print(f"  ✓  2θ = {center:7.3f}°   FWHM = {fwhm_deg:.4f}°   "
              f"β = {beta_rad:.5f} rad")

    except RuntimeError as e:
        print(f"  [fail] 2θ≈{two_theta[idx]:.2f}°: {e}")

n_good = len(fit_results)
print(f"\n  {n_good} peaks passed sanity check and will be used for analysis.")

if n_good == 0:
    raise RuntimeError(
        "No valid peaks after fitting. "
        "Adjust PEAK_HEIGHT_FRAC / PEAK_PROMINENCE / FWHM limits."
    )


# ═══════════════════════════════════════════════════════════════════
#  5.  DEBYE-SCHERRER  CRYSTALLITE SIZE
#
#  D = K λ / (β cosθ)
#  Average over all fitted peaks.
# ═══════════════════════════════════════════════════════════════════
print("\nSTEP 5  Debye-Scherrer analysis …")

D_list_DS = []
for r in fit_results:
    D = (K * LAMBDA_NM) / (r["beta_rad"] * np.cos(r["theta_rad"]))
    D_list_DS.append(D)
    print(f"  2θ = {r['two_theta_deg']:7.3f}°   D = {D:.3f} nm")

D_DS = float(np.mean(D_list_DS))
print(f"\n  ⟹  Mean D (Debye-Scherrer) = {D_DS:.3f} nm")


# ═══════════════════════════════════════════════════════════════════
#  5b. STOKES–WILSON MICRO-STRAIN  (per peak)
#
#  Equation:   ε = β / (4 tanθ)
#
#  This treats the ENTIRE peak broadening as strain — i.e. it gives
#  an UPPER BOUND on ε (size contribution is ignored here, unlike W-H).
#  It is valid and widely cited in the XRD literature for a quick
#  per-peak strain estimate alongside Debye-Scherrer size.
#
#  Note: β must be in radians, θ in radians → ε is dimensionless.
# ═══════════════════════════════════════════════════════════════════
print("\nSTEP 5b  Stokes–Wilson micro-strain (per peak) …")

eps_SW_list = []
for r in fit_results:
    beta  = r["beta_rad"]
    theta = r["theta_rad"]
    eps_sw = beta / (4.0 * np.tan(theta))   # dimensionless
    eps_SW_list.append(eps_sw)
    r["eps_SW"] = eps_sw                     # store on result dict
    print(f"  2θ = {r['two_theta_deg']:7.3f}°   ε (S-W) = {eps_sw*1e3:.4f} ×10⁻³")

eps_SW_mean = float(np.mean(eps_SW_list))
eps_SW_std  = float(np.std(eps_SW_list))
print(f"\n  ⟹  Mean ε (Stokes–Wilson) = {eps_SW_mean*1e3:.4f} ×10⁻³")
print(f"       Std  ε (Stokes–Wilson) = {eps_SW_std*1e3:.4f} ×10⁻³")
print( "       (upper bound — assumes all broadening is strain)")


# ═══════════════════════════════════════════════════════════════════
#  6.  WILLIAMSON-HALL ANALYSIS
#
#  Modified Williamson-Hall equation:
#    β cosθ = (K λ / D)  +  4 ε sinθ
#
#  Linear regression of  y = β cosθ  vs  x = 4 sinθ:
#    slope     → micro-strain  ε
#    intercept → K λ / D   ⟹   D = K λ / intercept
# ═══════════════════════════════════════════════════════════════════
print("\nSTEP 6  Williamson-Hall analysis …")

x_wh = np.array([4.0 * np.sin(r["theta_rad"]) for r in fit_results])
y_wh = np.array([r["beta_rad"] * np.cos(r["theta_rad"]) for r in fit_results])

slope_wh, intercept_wh, r_val, _, se_slope = linregress(x_wh, y_wh)

if intercept_wh <= 0:
    print("  WARNING: W-H intercept ≤ 0; D(W-H) not physically meaningful.")
    D_WH  = float("nan")
else:
    D_WH = (K * LAMBDA_NM) / intercept_wh

eps_WH = float(slope_wh)

print(f"  4 sinθ range:    {x_wh.min():.4f} – {x_wh.max():.4f}")
print(f"  β cosθ range:    {y_wh.min():.5f} – {y_wh.max():.5f} rad")
print(f"  Slope (ε):       {slope_wh:.6f}  ±  {se_slope:.6f}")
print(f"  Intercept (Kλ/D):{intercept_wh:.6f}")
print(f"  R²:              {r_val**2:.6f}")
print(f"  ⟹  D (Williamson-Hall) = {D_WH:.3f} nm")
print(f"  ⟹  ε (micro-strain)    = {eps_WH*1e3:.4f} ×10⁻³")


# ═══════════════════════════════════════════════════════════════════
#  7.  DISLOCATION DENSITY
#
#  δ = 1 / D²   (nm⁻²),  reported as  ×10⁻³ nm⁻²
# ═══════════════════════════════════════════════════════════════════
print("\nSTEP 7  Dislocation density …")
delta_DS = (1.0 / D_DS**2) * 1e3
delta_WH = (1.0 / D_WH**2) * 1e3 if not np.isnan(D_WH) else float("nan")
# For Stokes-Wilson we use D_DS (size from D-S) since S-W gives only strain
delta_SW = delta_DS   # same crystallite size basis as D-S
print(f"  δ (Debye-Scherrer)   = {delta_DS:.6f} ×10⁻³ nm⁻²")
print(f"  δ (Williamson-Hall)  = {delta_WH:.6f} ×10⁻³ nm⁻²")
print(f"  δ (Stokes–Wilson)    = {delta_SW:.6f} ×10⁻³ nm⁻²  (D from D-S)")


# ═══════════════════════════════════════════════════════════════════
#  8.  SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 75)
print("FINAL RESULTS SUMMARY")
print("=" * 75)

def _fmt(v, n=4):
    return "—" if (isinstance(v, float) and np.isnan(v)) else round(v, n)

summary = pd.DataFrame([
    {
        "Method":                              "Debye–Scherrer",
        "Crystallite size D (nm)":             _fmt(D_DS, 3),
        "Micro-strain ε (×10⁻³)":             "—",
        "Dislocation density δ (×10⁻³ nm⁻²)": _fmt(delta_DS, 6),
    },
    {
        "Method":                              "Stokes–Wilson",
        "Crystallite size D (nm)":             _fmt(D_DS, 3),   # D-S size
        "Micro-strain ε (×10⁻³)":             _fmt(eps_SW_mean * 1e3, 4),
        "Dislocation density δ (×10⁻³ nm⁻²)": _fmt(delta_SW, 6),
    },
    {
        "Method":                              "Williamson–Hall",
        "Crystallite size D (nm)":             _fmt(D_WH, 3),
        "Micro-strain ε (×10⁻³)":             _fmt(eps_WH * 1e3, 4),
        "Dislocation density δ (×10⁻³ nm⁻²)": _fmt(delta_WH, 6),
    },
])

print(summary.to_string(index=False))
print("=" * 75)
print("\nNote: Stokes–Wilson ε is an upper bound (all broadening attributed to")
print("      strain). Williamson–Hall separates size vs strain contributions.")
summary.to_csv("xrd_results_table.csv", index=False)
print("\nTable saved → xrd_results_table.csv")


# ═══════════════════════════════════════════════════════════════════
#  9.  PUBLICATION-QUALITY PLOTS
#
#  Style targets: Nature Materials / Elsevier / ACS Nano standard
#  ─ Font:        Times New Roman (serif), 10 pt body / 11 pt labels
#  ─ Size:        ~3.5 in per panel (single-column journal width)
#  ─ DPI:         300 (minimum for print submission)
#  ─ Colors:      Muted, colorblind-safe palette (no saturated primaries)
#  ─ Axes:        Full box frame, major + minor ticks inward on all sides
#  ─ Grid:        Off (journals rarely use grids in final figures)
#  ─ Titles:      None (figure captions go in the manuscript text)
#  ─ Legend:      Frameless or thin-framed, small serif font
#  ─ Lines:       Black raw data, dark navy smoothed, careful marker styles
# ═══════════════════════════════════════════════════════════════════
print("\nSTEP 8  Generating publication-quality plots …")

import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

# ── Global rcParams: apply journal style universally ─────────────
plt.rcParams.update({
    # Font — Times New Roman is standard in most journals
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset":   "stix",          # STIX = Times-compatible math
    "font.size":          10,
    "axes.titlesize":     10,
    "axes.labelsize":     11,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,

    # Axes frame: full box (all four spines visible)
    "axes.spines.top":    True,
    "axes.spines.right":  True,

    # Tick style: inward, both major and minor, all four sides
    "xtick.direction":    "in",
    "ytick.direction":    "in",
    "xtick.top":          True,
    "ytick.right":        True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.major.size":   5,
    "xtick.minor.size":   2.5,
    "ytick.major.size":   5,
    "ytick.minor.size":   2.5,
    "xtick.major.width":  0.8,
    "xtick.minor.width":  0.6,
    "ytick.major.width":  0.8,
    "ytick.minor.width":  0.6,

    # Line widths
    "axes.linewidth":     0.8,
    "lines.linewidth":    1.2,

    # No grid
    "axes.grid":          False,

    # Legend
    "legend.frameon":     True,
    "legend.framealpha":  0.9,
    "legend.edgecolor":   "0.7",
    "legend.handlelength": 2.0,

    # Figure background white
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",

    # Tight layout padding
    "figure.constrained_layout.use": False,
})

# ── Colorblind-safe, muted palette (Wong 2011 / IBM Carbon) ──────
C_RAW      = "#AAAAAA"      # light grey      — raw data
C_SMOOTH   = "#222222"      # near-black       — smoothed curve
C_PEAK     = "#CC3311"      # muted red        — peak markers
C_SCATTER  = "#004488"      # deep navy        — W-H data points
C_FIT      = "#000000"      # black dashed     — regression line
C_BAR_POS  = "#4477AA"      # steel blue       — S-W positive bars
C_BAR_NEG  = "#BB5566"      # muted rose       — S-W negative bars
C_MEAN     = "#222222"      # near-black       — mean line


# ════════════════════════════════════════════════════════════════════
#  FIGURE 1 — XRD Pattern
#  Width: 3.5 in (single journal column), height: 3.0 in
# ════════════════════════════════════════════════════════════════════
fig1, ax1 = plt.subplots(figsize=(3.5, 3.0))

y_max   = smoothed.max()
y_min   = smoothed.min()
y_range = y_max - y_min

# Raw data as very thin light-grey line
ax1.plot(two_theta, intensity,
         color=C_RAW, lw=0.5, zorder=1, label="As-measured")

# Smoothed curve — slightly thicker, near-black
ax1.plot(two_theta, smoothed,
         color=C_SMOOTH, lw=1.0, zorder=2,
         label=f"Adjacent avg. ($n$={AA_WINDOW})")

# Peak markers + vertical tick lines + 2θ text labels
for r in fit_results:
    p2t = r["two_theta_deg"]
    yi  = smoothed[np.argmin(np.abs(two_theta - p2t))]

    # Small filled circle at peak apex
    ax1.plot(p2t, yi, marker="o", ms=3.0, color=C_PEAK,
             mec=C_PEAK, mew=0.5, zorder=5, ls="none")

    # Short vertical tick line upward from apex
    tick_top = yi + 0.05 * y_range
    ax1.plot([p2t, p2t], [yi, tick_top],
             color=C_PEAK, lw=0.7, zorder=4)

    # 2θ label, rotated 90°, anchored at tick_top
    ax1.text(p2t, tick_top + 0.005 * y_range,
             f"{p2t:.2f}",
             ha="center", va="bottom",
             rotation=90,
             fontsize=5.5,
             color=C_PEAK,
             fontfamily="serif")

# Axis labels — use LaTeX-style degree symbol
ax1.set_xlabel(r"2$\theta$ (degrees)")
ax1.set_ylabel("Intensity (arb. units)")

# x-axis: 10° major ticks, 5° minor
ax1.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax1.xaxis.set_minor_locator(ticker.MultipleLocator(5))
ax1.yaxis.set_major_locator(ticker.MaxNLocator(5, integer=True))
ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

# Headroom for labels; bottom at zero or just below min
ax1.set_ylim(bottom=y_min - 0.03 * y_range,
             top=y_max  + 0.30 * y_range)
ax1.set_xlim(two_theta[0] - 1, two_theta[-1] + 1)

legend1 = ax1.legend(
    loc="upper right",
    handlelength=1.5,
    borderpad=0.5,
    labelspacing=0.3,
)

fig1.tight_layout(pad=0.4)
fig1.savefig("fig1_xrd_pattern.png",    dpi=300, bbox_inches="tight")
fig1.savefig("fig1_xrd_pattern.pdf",    bbox_inches="tight")
print("  Saved fig1_xrd_pattern.png / .pdf")


# ════════════════════════════════════════════════════════════════════
#  FIGURE 2 — Williamson–Hall Plot
#  Width: 3.0 in, height: 3.0 in  (square, common for scatter plots)
# ════════════════════════════════════════════════════════════════════
fig2, ax2 = plt.subplots(figsize=(3.0, 3.0))

# Extended fit line with a small margin
x_margin = 0.05 * (x_wh.max() - x_wh.min())
x_line   = np.linspace(x_wh.min() - x_margin, x_wh.max() + x_margin, 400)
y_line   = slope_wh * x_line + intercept_wh

# Dashed regression line first (so it sits behind markers)
ax2.plot(x_line, y_line,
         color=C_FIT, lw=0.9, ls="--", zorder=2,
         label=(rf"Linear fit ($R^2$ = {r_val**2:.4f})" "\n"
                rf"$D$ = {D_WH:.2f} nm, "
                rf"$\varepsilon$ = {eps_WH*1e3:.3f} $\times10^{{-3}}$"))

# Data points — filled circles, navy
ax2.scatter(x_wh, y_wh,
            s=28, color=C_SCATTER, marker="o",
            edgecolors=C_SCATTER, linewidths=0.5,
            zorder=3, label="Experimental")

# ── 2θ label next to every scatter point ─────────────────────────
# Strategy: place label above/below the regression line so it never
# sits on top of the fit line.  Points above the line → label above;
# points below → label below.  A fixed horizontal nudge avoids the dot.
y_fit_at_x = slope_wh * x_wh + intercept_wh   # predicted y at each x

for xi, yi, r in zip(x_wh, y_wh, fit_results):
    above = yi >= (slope_wh * xi + intercept_wh)
    dy =  6 if above else -8          # pt offset: up if above line, down if below
    dx =  4                           # always nudge right to clear the marker

    ax2.annotate(
        f"{r['two_theta_deg']:.2f}°",
        xy         = (xi, yi),
        xytext     = (dx, dy),
        textcoords = "offset points",
        fontsize   = 7,
        color      = "#333333",
        fontfamily = "serif",
        va         = "bottom" if above else "top",
        ha         = "left",
    )

ax2.set_xlabel(r"4 sin$\theta$")
ax2.set_ylabel(r"$\beta$ cos$\theta$ (rad)")

# Auto-tick with minor subdivisions
ax2.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
ax2.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

# Padding around data
xpad = 0.12 * (x_wh.max() - x_wh.min())
ypad = 0.12 * (y_wh.max() - y_wh.min())
ax2.set_xlim(x_wh.min() - xpad, x_wh.max() + xpad)
ax2.set_ylim(y_wh.min() - ypad, y_wh.max() + ypad)

ax2.legend(loc="upper left",
           handlelength=1.5,
           borderpad=0.5,
           labelspacing=0.3)

fig2.tight_layout(pad=0.4)
fig2.savefig("fig2_williamson_hall.png", dpi=300, bbox_inches="tight")
fig2.savefig("fig2_williamson_hall.pdf", bbox_inches="tight")
print("  Saved fig2_williamson_hall.png / .pdf")


# ════════════════════════════════════════════════════════════════════
#  FIGURE 3 — Stokes–Wilson Micro-Strain per Peak
#  Width: 3.5 in, height: 2.8 in
#  Style: scatter/lollipop (more elegant than bars for publications)
# ════════════════════════════════════════════════════════════════════
fig3, ax3 = plt.subplots(figsize=(3.5, 2.8))

sw_2theta = np.array([r["two_theta_deg"] for r in fit_results])
sw_eps    = np.array([r["eps_SW"] * 1e3  for r in fit_results])

# Lollipop chart: vertical stem + filled circle at tip
for xi, ei in zip(sw_2theta, sw_eps):
    col = C_BAR_POS if ei >= 0 else C_BAR_NEG
    ax3.plot([xi, xi], [0, ei],
             color=col, lw=0.9, zorder=2)
    ax3.plot(xi, ei,
             marker="o", ms=4.0,
             color=col, mec=col, mew=0.5,
             zorder=3, ls="none")

# Mean line
ax3.axhline(eps_SW_mean * 1e3,
            color=C_MEAN, lw=0.9, ls=(0, (4, 2)),   # long-dash
            zorder=4,
            label=rf"Mean $\varepsilon$ = {eps_SW_mean*1e3:.3f} $\times10^{{-3}}$")

# Zero baseline
ax3.axhline(0, color="black", lw=0.6, zorder=1)

# x-ticks at actual peak positions, rotated
ax3.set_xticks(sw_2theta)
ax3.set_xticklabels([f"{v:.1f}" for v in sw_2theta],
                    rotation=60, ha="right", fontsize=7)

ax3.set_xlabel(r"2$\theta$ (degrees)")
ax3.set_ylabel(r"Micro-strain $\varepsilon$ ($\times10^{-3}$)")

ax3.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
# Minor ticks on x are not meaningful here (categorical-like spacing)
ax3.xaxis.set_minor_locator(ticker.NullLocator())

# y-axis padding
eps_span = sw_eps.max() - sw_eps.min()
ax3.set_ylim(sw_eps.min() - 0.12 * eps_span,
             sw_eps.max() + 0.18 * eps_span)
ax3.set_xlim(sw_2theta[0]  - 2,
             sw_2theta[-1] + 2)

ax3.legend(loc="upper right",
           handlelength=1.5,
           borderpad=0.5,
           labelspacing=0.3)

fig3.tight_layout(pad=0.4)
fig3.savefig("fig3_stokes_wilson.png", dpi=300, bbox_inches="tight")
fig3.savefig("fig3_stokes_wilson.pdf", bbox_inches="tight")
print("  Saved fig3_stokes_wilson.png / .pdf")


# ════════════════════════════════════════════════════════════════════
#  COMBINED PANEL FIGURE  (for convenience / supplementary)
#  Three panels side-by-side, 7.2 in wide (full journal page width)
# ════════════════════════════════════════════════════════════════════
fig_all, axes_all = plt.subplots(1, 3, figsize=(7.2, 2.8))
fig_all.subplots_adjust(wspace=0.45)

# ── Panel (a): XRD pattern ───────────────────────────────────────
aa = axes_all[0]
aa.plot(two_theta, intensity, color=C_RAW,    lw=0.4, zorder=1)
aa.plot(two_theta, smoothed,  color=C_SMOOTH, lw=0.9, zorder=2,
        label=f"Smoothed")

for r in fit_results:
    p2t = r["two_theta_deg"]
    yi  = smoothed[np.argmin(np.abs(two_theta - p2t))]
    tick_top = yi + 0.05 * y_range
    aa.plot(p2t, yi,         marker="o", ms=2.2, color=C_PEAK,
            mec=C_PEAK, ls="none", zorder=5)
    aa.plot([p2t, p2t], [yi, tick_top], color=C_PEAK, lw=0.6, zorder=4)
    aa.text(p2t, tick_top + 0.005 * y_range,
            f"{p2t:.1f}", ha="center", va="bottom",
            rotation=90, fontsize=4.2, color=C_PEAK, fontfamily="serif")

aa.set_xlabel(r"2$\theta$ (deg)", fontsize=8)
aa.set_ylabel("Intensity (arb. units)", fontsize=8)
aa.tick_params(labelsize=7)
aa.xaxis.set_major_locator(ticker.MultipleLocator(10))
aa.xaxis.set_minor_locator(ticker.MultipleLocator(5))
aa.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
aa.set_ylim(bottom=y_min - 0.03*y_range, top=y_max + 0.30*y_range)
aa.set_xlim(two_theta[0]-1, two_theta[-1]+1)
aa.text(0.04, 0.96, "(a)", transform=aa.transAxes,
        fontsize=9, fontweight="bold", va="top", fontfamily="serif")

# ── Panel (b): W-H plot ──────────────────────────────────────────
ab = axes_all[1]
ab.plot(x_line, y_line, color=C_FIT, lw=0.8, ls="--", zorder=2)
ab.scatter(x_wh, y_wh, s=18, color=C_SCATTER,
           edgecolors=C_SCATTER, lw=0.4, zorder=3)

# 2θ labels on every point — above/below the fit line to avoid overlap
for xi, yi, r in zip(x_wh, y_wh, fit_results):
    above = yi >= (slope_wh * xi + intercept_wh)
    dy = 5 if above else -7
    ab.annotate(
        f"{r['two_theta_deg']:.1f}°",
        xy         = (xi, yi),
        xytext     = (3, dy),
        textcoords = "offset points",
        fontsize   = 5.5,
        color      = "#333333",
        fontfamily = "serif",
        va         = "bottom" if above else "top",
        ha         = "left",
    )
# Compact in-plot text box instead of legend
textstr = (rf"$R^2$ = {r_val**2:.4f}" "\n"
           rf"$D$ = {D_WH:.1f} nm" "\n"
           rf"$\varepsilon$ = {eps_WH*1e3:.3f}$\times10^{{-3}}$")
ab.text(0.96, 0.06, textstr,
        transform=ab.transAxes,
        fontsize=6.5, va="bottom", ha="right",
        fontfamily="serif",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", lw=0.6))
ab.set_xlabel(r"4 sin$\theta$", fontsize=8)
ab.set_ylabel(r"$\beta$ cos$\theta$ (rad)", fontsize=8)
ab.tick_params(labelsize=7)
ab.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
ab.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
ab.set_xlim(x_wh.min()-xpad, x_wh.max()+xpad)
ab.set_ylim(y_wh.min()-ypad, y_wh.max()+ypad)
ab.text(0.04, 0.96, "(b)", transform=ab.transAxes,
        fontsize=9, fontweight="bold", va="top", fontfamily="serif")

# ── Panel (c): S-W lollipop ──────────────────────────────────────
ac = axes_all[2]
for xi, ei in zip(sw_2theta, sw_eps):
    col = C_BAR_POS if ei >= 0 else C_BAR_NEG
    ac.plot([xi, xi], [0, ei], color=col, lw=0.8, zorder=2)
    ac.plot(xi, ei, marker="o", ms=3.0, color=col,
            mec=col, mew=0.4, zorder=3, ls="none")
ac.axhline(eps_SW_mean*1e3, color=C_MEAN, lw=0.8,
           ls=(0, (4, 2)), zorder=4)
ac.axhline(0, color="black", lw=0.5, zorder=1)
ac.set_xticks(sw_2theta)
ac.set_xticklabels([f"{v:.0f}" for v in sw_2theta],
                   rotation=60, ha="right", fontsize=5.5)
ac.set_xlabel(r"2$\theta$ (deg)", fontsize=8)
ac.set_ylabel(r"$\varepsilon$ ($\times10^{-3}$)", fontsize=8)
ac.tick_params(labelsize=7)
ac.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
ac.xaxis.set_minor_locator(ticker.NullLocator())
ac.set_ylim(sw_eps.min()-0.12*eps_span, sw_eps.max()+0.18*eps_span)
ac.set_xlim(sw_2theta[0]-2, sw_2theta[-1]+2)
ac.text(0.96, 0.96,
        rf"$\bar{{\varepsilon}}$ = {eps_SW_mean*1e3:.2f}$\times10^{{-3}}$",
        transform=ac.transAxes,
        fontsize=6.5, va="top", ha="right", fontfamily="serif")
ac.text(0.04, 0.96, "(c)", transform=ac.transAxes,
        fontsize=9, fontweight="bold", va="top", fontfamily="serif")

fig_all.savefig("fig_combined.png", dpi=300, bbox_inches="tight")
fig_all.savefig("fig_combined.pdf",           bbox_inches="tight")
print("  Saved fig_combined.png / .pdf")
print("\nDone.\n")
print("Output files:")
print("  fig1_xrd_pattern.png/pdf     — XRD pattern (single-column)")
print("  fig2_williamson_hall.png/pdf — W-H plot    (single-column)")
print("  fig3_stokes_wilson.png/pdf   — S-W strains (single-column)")
print("  fig_combined.png/pdf         — All three panels (full-width)")
