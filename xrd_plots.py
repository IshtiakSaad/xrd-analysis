"""
xrd_plots.py
============
All figure-generation functions.
Each returns a matplotlib Figure object — Streamlit renders it with
st.pyplot(fig); savefig() writes PNG/PDF for the download zip.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from scipy.special import voigt_profile


# ──────────────────────────────────────────────────────────────────
#  SHARED STYLE
# ──────────────────────────────────────────────────────────────────
def get_pub_rc(is_dark=False):
    base_rc = {
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
        "axes.grid":            False,
        "legend.frameon":       True,
        "legend.framealpha":    0.92,
    }
    if is_dark:
        base_rc.update({
            "figure.facecolor": "none",
            "axes.facecolor":   "none",
            "text.color":       "#E0E0E0",
            "axes.labelcolor":  "#E0E0E0",
            "xtick.color":      "#E0E0E0",
            "ytick.color":      "#E0E0E0",
            "axes.edgecolor":   "#555555",
            "legend.edgecolor": "#555555",
            "legend.facecolor": "#1E1E1E",
        })
    else:
        base_rc.update({
            "figure.facecolor": "white",
            "axes.facecolor":   "white",
            "text.color":       "#1A1A1A",
            "axes.labelcolor":  "#1A1A1A",
            "xtick.color":      "#1A1A1A",
            "ytick.color":      "#1A1A1A",
            "axes.edgecolor":   "#333333",
            "legend.edgecolor": "#CCCCCC",
            "legend.facecolor": "white",
        })
    return base_rc

def get_colors(is_dark=False):
    if is_dark:
        return {
            "RAW":     "#666666",
            "SMOOTH":  "#FF4B4B",
            "PEAK":    "#FF4B4B",
            "SCATTER": "#0068C9",
            "FIT":     "#AAAAAA",
            "POS":     "#0068C9",
            "NEG":     "#FF4B4B",
            "MEAN":    "#AAAAAA",
            "TEXT":    "#E0E0E0",
            "BBOX_FC": "#1E1E1E",
            "BBOX_EC": "#555555",
            "PEAK_PALETTE": ["#0068C9", "#FF4B4B", "#DDAA33", "#E0E0E0",
                            "#29B09D", "#FF2B2B", "#7D3538", "#F0F2F6"]
        }
    else:
        return {
            "RAW":     "#AAAAAA",
            "SMOOTH":  "#222222",
            "PEAK":    "#CC3311",
            "SCATTER": "#004488",
            "FIT":     "#000000",
            "POS":     "#4477AA",
            "NEG":     "#BB5566",
            "MEAN":    "#222222",
            "TEXT":    "#333333",
            "BBOX_FC": "white",
            "BBOX_EC": "0.7",
            "PEAK_PALETTE": ["#004488", "#BB5566", "#DDAA33", "#000000",
                            "#117733", "#882255", "#44AA99", "#999933"]
        }

# Initial defaults (light)
C = get_colors(False)
PEAK_COLORS = C["PEAK_PALETTE"]


def _voigt_model(x, amp, center, sigma, gamma, bg0, bg1):
    return amp * voigt_profile(x - center, sigma, gamma) + bg0 + bg1 * (x - center)


# ──────────────────────────────────────────────────────────────────
#  FIGURE 1 — XRD Pattern
# ──────────────────────────────────────────────────────────────────
def fig_xrd_pattern(two_theta, intensity, y_aa, fit_results,
                    aa_window=20, is_dark=False) -> plt.Figure:
    c = get_colors(is_dark)
    with plt.rc_context(get_pub_rc(is_dark)):
        fig, ax = plt.subplots(figsize=(3.5, 3.0))

        y_max   = y_aa.max()
        y_min   = y_aa.min()
        y_range = y_max - y_min

        ax.plot(two_theta, intensity, color=c["RAW"],    lw=0.5, zorder=1,
                label="As-measured")
        ax.plot(two_theta, y_aa,      color=c["SMOOTH"], lw=1.0, zorder=2,
                label=f"Adjacent avg. ($n$={aa_window})")

        for r in fit_results:
            p2t      = r["two_theta_deg"]
            yi       = y_aa[np.argmin(np.abs(two_theta - p2t))]
            tick_top = yi + 0.05 * y_range

            ax.plot(p2t, yi, marker="o", ms=3.0, color=c["PEAK"],
                    mec=c["PEAK"], mew=0.5, zorder=5, ls="none")
            ax.plot([p2t, p2t], [yi, tick_top],
                    color=c["PEAK"], lw=0.7, zorder=4)
            ax.text(p2t, tick_top + 0.005 * y_range,
                    f"{p2t:.2f}",
                    ha="center", va="bottom", rotation=90,
                    fontsize=5.5, color=c["PEAK"], fontfamily="serif")

        ax.set_xlabel(r"2$\theta$ (degrees)")
        ax.set_ylabel("Intensity (arb. units)")
        ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(5, integer=True))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        ax.set_ylim(bottom=y_min - 0.03 * y_range,
                    top=y_max    + 0.30 * y_range)
        ax.set_xlim(two_theta[0] - 1, two_theta[-1] + 1)
        ax.legend(loc="upper right", handlelength=1.5,
                  borderpad=0.5, labelspacing=0.3)

        fig.tight_layout(pad=0.4)
    return fig


# ──────────────────────────────────────────────────────────────────
#  FIGURE 2 — Williamson–Hall Plot
# ──────────────────────────────────────────────────────────────────
def fig_williamson_hall(fit_results, wh: dict, is_dark=False) -> plt.Figure:
    x_wh = wh["x"]
    y_wh = wh["y"]
    slope, intercept = wh["slope"], wh["intercept"]
    D_WH, eps_WH, r_sq = wh["D_nm"], wh["eps"], wh["r_sq"]
    c = get_colors(is_dark)

    # Safe formatting for potentially NaN values
    r_sq_str = f"{r_sq:.4f}" if np.isfinite(r_sq) else "—"
    D_WH_str = f"{D_WH:.2f}" if np.isfinite(D_WH) else "—"

    with plt.rc_context(get_pub_rc(is_dark)):
        fig, ax = plt.subplots(figsize=(3.0, 3.0))

        x_range = x_wh.max() - x_wh.min() if len(x_wh) > 1 else 1.0
        x_margin = max(0.05 * x_range, 0.01)
        x_line   = np.linspace(x_wh.min() - x_margin,
                               x_wh.max() + x_margin, 400)
        y_line   = slope * x_line + intercept

        ax.plot(x_line, y_line, color=c["FIT"], lw=0.9, ls="--", zorder=2,
                label=(rf"Linear fit ($R^2$ = {r_sq_str})" "\n"
                       rf"$D$ = {D_WH_str} nm, "
                       rf"$\varepsilon$ = {eps_WH*1e3:.3f}$\times10^{{-3}}$"))

        ax.scatter(x_wh, y_wh, s=28, color=c["SCATTER"],
                   edgecolors=c["SCATTER"], linewidths=0.5,
                   zorder=3, label="Experimental")

        for xi, yi, r in zip(x_wh, y_wh, fit_results):
            above = yi >= (slope * xi + intercept)
            dy    =  6 if above else -8
            ax.annotate(
                f"{r['two_theta_deg']:.2f}°",
                xy=(xi, yi), xytext=(4, dy),
                textcoords="offset points",
                fontsize=7, color=c["TEXT"], fontfamily="serif",
                va="bottom" if above else "top", ha="left",
            )

        ax.set_xlabel(r"4 sin$\theta$")
        ax.set_ylabel(r"$\beta$ cos$\theta$ (rad)")
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

        xpad = max(0.12 * x_range, 0.01)
        y_range = y_wh.max() - y_wh.min() if len(y_wh) > 1 else abs(y_wh[0]) * 0.2 if len(y_wh) == 1 else 1.0
        ypad = max(0.12 * y_range, 0.001)
        ax.set_xlim(x_wh.min() - xpad, x_wh.max() + xpad)
        ax.set_ylim(y_wh.min() - ypad, y_wh.max() + ypad)
        ax.legend(loc="upper left", handlelength=1.5,
                  borderpad=0.5, labelspacing=0.3)

        fig.tight_layout(pad=0.4)
    return fig


# ──────────────────────────────────────────────────────────────────
#  FIGURE 3 — Stokes–Wilson Micro-Strain per Peak
# ──────────────────────────────────────────────────────────────────
def fig_stokes_wilson(fit_results, eps_SW_list, eps_SW_mean, is_dark=False) -> plt.Figure:
    sw_2t  = np.array([r["two_theta_deg"] for r in fit_results])
    # Filter out NaN values
    raw_eps = np.array(eps_SW_list) * 1e3
    valid_mask = np.isfinite(raw_eps)
    sw_eps = np.where(valid_mask, raw_eps, 0.0)
    c = get_colors(is_dark)

    # Safe mean formatting
    mean_str = f"{eps_SW_mean*1e3:.3f}" if np.isfinite(eps_SW_mean) else "—"

    with plt.rc_context(get_pub_rc(is_dark)):
        fig, ax = plt.subplots(figsize=(3.5, 2.8))

        eps_span = sw_eps.max() - sw_eps.min() if len(sw_eps) > 1 else max(abs(sw_eps[0]), 0.1)
        if eps_span < 1e-10:
            eps_span = max(abs(sw_eps.max()), 0.1)  # fallback

        for xi, ei, valid in zip(sw_2t, sw_eps, valid_mask):
            if not valid:
                continue
            col = c["POS"] if ei >= 0 else c["NEG"]
            ax.plot([xi, xi], [0, ei], color=col, lw=0.9, zorder=2)
            ax.plot(xi, ei, marker="o", ms=4.0,
                    color=col, mec=col, mew=0.5, zorder=3, ls="none")

        if np.isfinite(eps_SW_mean):
            ax.axhline(eps_SW_mean * 1e3, color=c["MEAN"], lw=0.9,
                       ls=(0, (4, 2)), zorder=4,
                       label=(rf"Mean $\varepsilon$ = "
                              rf"{mean_str}$\times10^{{-3}}$"))
        ax.axhline(0, color=c["TEXT"], lw=0.6, zorder=1, alpha=0.5)

        ax.set_xticks(sw_2t)
        ax.set_xticklabels([f"{v:.1f}" for v in sw_2t],
                           rotation=60, ha="right", fontsize=7)
        ax.set_xlabel(r"2$\theta$ (degrees)")
        ax.set_ylabel(r"Micro-strain $\varepsilon$ ($\times10^{-3}$)")
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        ax.xaxis.set_minor_locator(ticker.NullLocator())
        ax.set_ylim(sw_eps.min() - 0.12 * eps_span,
                    sw_eps.max() + 0.20 * eps_span)
        x_range = sw_2t[-1] - sw_2t[0] if len(sw_2t) > 1 else 10.0
        ax.set_xlim(sw_2t[0] - max(2, 0.05 * x_range),
                    sw_2t[-1] + max(2, 0.05 * x_range))
        ax.legend(loc="upper right", handlelength=1.5,
                  borderpad=0.5, labelspacing=0.3)

        fig.tight_layout(pad=0.4)
    return fig


# ──────────────────────────────────────────────────────────────────
#  FIGURE 4 — Combined 3-panel
# ──────────────────────────────────────────────────────────────────
def fig_combined(two_theta, intensity, y_aa, fit_results,
                 wh, eps_SW_list, eps_SW_mean, aa_window=20, is_dark=False) -> plt.Figure:
    x_wh     = wh["x"]
    y_wh     = wh["y"]
    slope    = wh["slope"]
    intercept= wh["intercept"]
    D_WH     = wh["D_nm"]
    eps_WH   = wh["eps"]
    r_sq     = wh["r_sq"]

    sw_2t  = np.array([r["two_theta_deg"] for r in fit_results])
    raw_eps = np.array(eps_SW_list) * 1e3
    valid_mask = np.isfinite(raw_eps)
    sw_eps = np.where(valid_mask, raw_eps, 0.0)
    eps_span = sw_eps.max() - sw_eps.min() if len(sw_eps) > 1 else max(abs(sw_eps[0]), 0.1)
    if eps_span < 1e-10:
        eps_span = max(abs(sw_eps.max()), 0.1)

    y_max   = y_aa.max()
    y_min   = y_aa.min()
    y_range = max(y_max - y_min, 1.0)

    wh_x_range = x_wh.max() - x_wh.min() if len(x_wh) > 1 else 1.0
    xpad = max(0.12 * wh_x_range, 0.01)
    wh_y_range = y_wh.max() - y_wh.min() if len(y_wh) > 1 else abs(y_wh[0]) * 0.2 if len(y_wh) == 1 else 1.0
    ypad = max(0.12 * wh_y_range, 0.001)
    x_line = np.linspace(x_wh.min() - xpad, x_wh.max() + xpad, 400)
    y_line = slope * x_line + intercept

    c = get_colors(is_dark)

    # Safe formatting for NaN values
    r_sq_str = f"{r_sq:.4f}" if np.isfinite(r_sq) else "—"
    D_WH_str = f"{D_WH:.1f}" if np.isfinite(D_WH) else "—"
    mean_str = f"{eps_SW_mean*1e3:.2f}" if np.isfinite(eps_SW_mean) else "—"

    with plt.rc_context(get_pub_rc(is_dark)):
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8))
        fig.subplots_adjust(wspace=0.45)

        # (a) XRD pattern
        aa = axes[0]
        aa.plot(two_theta, intensity, color=c["RAW"],    lw=0.4, zorder=1)
        aa.plot(two_theta, y_aa,      color=c["SMOOTH"], lw=0.9, zorder=2)
        for r in fit_results:
            p2t = r["two_theta_deg"]
            yi  = y_aa[np.argmin(np.abs(two_theta - p2t))]
            tick_top = yi + 0.05 * y_range
            aa.plot(p2t, yi, marker="o", ms=2.2, color=c["PEAK"],
                    mec=c["PEAK"], ls="none", zorder=5)
            aa.plot([p2t, p2t], [yi, tick_top],
                    color=c["PEAK"], lw=0.6, zorder=4)
            aa.text(p2t, tick_top + 0.005 * y_range,
                    f"{p2t:.1f}", ha="center", va="bottom",
                    rotation=90, fontsize=4.2,
                    color=c["PEAK"], fontfamily="serif")
        aa.set_xlabel(r"2$\theta$ (deg)", fontsize=8)
        aa.set_ylabel("Intensity (arb. units)", fontsize=8)
        aa.tick_params(labelsize=7)
        aa.xaxis.set_major_locator(ticker.MultipleLocator(10))
        aa.xaxis.set_minor_locator(ticker.MultipleLocator(5))
        aa.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        aa.set_ylim(bottom=y_min - 0.03 * y_range,
                    top=y_max    + 0.30 * y_range)
        aa.set_xlim(two_theta[0] - 1, two_theta[-1] + 1)
        aa.text(0.04, 0.96, "(a)", transform=aa.transAxes,
                fontsize=9, fontweight="bold", va="top", fontfamily="serif")

        # (b) W-H
        ab = axes[1]
        ab.plot(x_line, y_line, color=c["FIT"], lw=0.8, ls="--", zorder=2)
        ab.scatter(x_wh, y_wh, s=18, color=c["SCATTER"],
                   edgecolors=c["SCATTER"], lw=0.4, zorder=3)
        for xi, yi, r in zip(x_wh, y_wh, fit_results):
            above = yi >= (slope * xi + intercept)
            dy = 5 if above else -7
            ab.annotate(f"{r['two_theta_deg']:.1f}°",
                        xy=(xi, yi), xytext=(3, dy),
                        textcoords="offset points",
                        fontsize=5.5, color=c["TEXT"],
                        fontfamily="serif",
                        va="bottom" if above else "top", ha="left")
        textstr = (rf"$R^2$ = {r_sq_str}" "\n"
                   rf"$D$ = {D_WH_str} nm" "\n"
                   rf"$\varepsilon$ = {eps_WH*1e3:.3f}$\times10^{{-3}}$")
        ab.text(0.96, 0.06, textstr, transform=ab.transAxes,
                fontsize=6.5, va="bottom", ha="right", fontfamily="serif",
                bbox=dict(boxstyle="round,pad=0.3", fc=c["BBOX_FC"],
                          ec=c["BBOX_EC"], lw=0.6))
        ab.set_xlabel(r"4 sin$\theta$", fontsize=8)
        ab.set_ylabel(r"$\beta$ cos$\theta$ (rad)", fontsize=8)
        ab.tick_params(labelsize=7)
        ab.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        ab.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        ab.set_xlim(x_wh.min() - xpad, x_wh.max() + xpad)
        ab.set_ylim(y_wh.min() - ypad, y_wh.max() + ypad)
        ab.text(0.04, 0.96, "(b)", transform=ab.transAxes,
                fontsize=9, fontweight="bold", va="top", fontfamily="serif")

        # (c) S-W
        ac = axes[2]
        for xi, ei, valid in zip(sw_2t, sw_eps, valid_mask):
            if not valid:
                continue
            col = c["POS"] if ei >= 0 else c["NEG"]
            ac.plot([xi, xi], [0, ei], color=col, lw=0.8, zorder=2)
            ac.plot(xi, ei, marker="o", ms=3.0, color=col,
                    mec=col, mew=0.4, zorder=3, ls="none")
        if np.isfinite(eps_SW_mean):
            ac.axhline(eps_SW_mean * 1e3, color=c["MEAN"], lw=0.8,
                       ls=(0, (4, 2)), zorder=4)
        ac.axhline(0, color=c["TEXT"], lw=0.5, zorder=1, alpha=0.5)
        ac.set_xticks(sw_2t)
        ac.set_xticklabels([f"{v:.0f}" for v in sw_2t],
                           rotation=60, ha="right", fontsize=5.5)
        ac.set_xlabel(r"2$\theta$ (deg)", fontsize=8)
        ac.set_ylabel(r"$\varepsilon$ ($\times10^{-3}$)", fontsize=8)
        ac.tick_params(labelsize=7)
        ac.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        ac.xaxis.set_minor_locator(ticker.NullLocator())
        ac.set_ylim(sw_eps.min() - 0.12 * eps_span,
                    sw_eps.max() + 0.18 * eps_span)
        sw_x_range = sw_2t[-1] - sw_2t[0] if len(sw_2t) > 1 else 10.0
        ac.set_xlim(sw_2t[0] - max(2, 0.05 * sw_x_range),
                    sw_2t[-1] + max(2, 0.05 * sw_x_range))
        ac.text(0.96, 0.96,
                rf"$\bar{{\varepsilon}}$ = {mean_str}$\times10^{{-3}}$",
                transform=ac.transAxes, fontsize=6.5, va="top", ha="right",
                fontfamily="serif")
        ac.text(0.04, 0.96, "(c)", transform=ac.transAxes,
                fontsize=9, fontweight="bold", va="top", fontfamily="serif")

    return fig


# ──────────────────────────────────────────────────────────────────
#  FIGURE 5 — Peak Shift Comparison (multi-sample)
# ──────────────────────────────────────────────────────────────────
def fig_peak_shift(records: list[dict], ref_idx: int = 0,
                   zoom_margin: float = 1.2,
                   stack_frac: float = 0.15, is_dark=False) -> plt.Figure:
    n = len(records)
    c = get_colors(is_dark)
    colors = (c["PEAK_PALETTE"] * ((n // len(c["PEAK_PALETTE"])) + 1))[:n]

    ref_center = records[ref_idx]["center"]

    global_max  = max(r["y_aa"].max() for r in records)
    offset_step = stack_frac * global_max
    all_centers = [r["center"] for r in records]
    zoom_lo     = min(all_centers) - zoom_margin
    zoom_hi     = max(all_centers) + zoom_margin

    with plt.rc_context(get_pub_rc(is_dark)):
        fig, (ax_full, ax_zoom) = plt.subplots(
            1, 2, figsize=(7.2, 3.2),
            gridspec_kw={"width_ratios": [2.8, 1.8]},
        )
        fig.subplots_adjust(wspace=0.38)

        # (a) full-range stacked
        for i, r in enumerate(records):
            offset = i * offset_step
            ax_full.plot(r["x"], r["y_aa"] + offset,
                         color=colors[i], lw=0.9, label=r["name"], zorder=n - i)
            yt = r["y_aa"][r["dom_idx"]] + offset
            ax_full.plot(r["center"], yt, marker="|", ms=7, mew=1.2,
                         color=colors[i], ls="none", zorder=n + 1)

        ax_full.set_xlabel(r"2$\theta$ (degrees)")
        ax_full.set_ylabel("Intensity (arb. units)")
        ax_full.xaxis.set_major_locator(ticker.MultipleLocator(10))
        ax_full.xaxis.set_minor_locator(ticker.MultipleLocator(5))
        ax_full.yaxis.set_major_locator(ticker.NullLocator())
        ax_full.set_xlim(records[0]["x"][0] - 0.5, records[0]["x"][-1] + 0.5)
        ax_full.set_ylim(-0.02 * global_max,
                         global_max + (n + 0.3) * offset_step)
        ax_full.legend(loc="upper right", handlelength=1.6,
                       borderpad=0.4, labelspacing=0.25, fontsize=8.5)
        ax_full.text(0.03, 0.97, "(a)", transform=ax_full.transAxes,
                     fontsize=9, fontweight="bold", va="top", fontfamily="serif")

        # (b) zoomed normalised
        for i, r in enumerate(records):
            mask   = (r["x"] >= zoom_lo - 1) & (r["x"] <= zoom_hi + 1)
            x_z    = r["x"][mask]
            y_z    = r["y_aa"][mask]
            y_norm = y_z / y_z.max()
            offset = i * 0.28

            ax_zoom.plot(x_z, y_norm + offset,
                         color=colors[i], lw=1.0, zorder=n - i)
            peak_y = 1.0 + offset
            ax_zoom.plot([r["center"], r["center"]],
                         [offset, peak_y + 0.04],
                         color=colors[i], lw=0.7, ls="--", zorder=n + 1)

            ha_side = "left" if i % 2 == 0 else "right"
            dx_pt   = 3 if i % 2 == 0 else -3
            ax_zoom.annotate(
                f"{r['center']:.3f}°",
                xy=(r["center"], peak_y + 0.06),
                xytext=(dx_pt, 2), textcoords="offset points",
                fontsize=7.5, fontfamily="serif", color=colors[i],
                ha=ha_side, va="bottom", fontweight="semibold",
            )
            if i != ref_idx:
                shift = r["center"] - ref_center
                sign  = "+" if shift > 0 else ""
                ax_zoom.annotate(
                    f"$\\Delta$2$\\theta$ = {sign}{shift:.3f}°",
                    xy=(r["center"], offset + 0.5),
                    xytext=(4, 0), textcoords="offset points",
                    fontsize=6.5, fontfamily="serif", color=colors[i],
                    va="center", ha="left",
                )

        ax_zoom.set_xlabel(r"2$\theta$ (degrees)")
        ax_zoom.set_ylabel("Norm. intensity (arb. units)")
        ax_zoom.set_xlim(zoom_lo, zoom_hi)
        ax_zoom.yaxis.set_major_locator(ticker.NullLocator())
        ax_zoom.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax_zoom.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
        ax_zoom.text(0.04, 0.97, "(b)", transform=ax_zoom.transAxes,
                     fontsize=9, fontweight="bold", va="top", fontfamily="serif")
        ax_zoom.text(0.97, 0.03,
                     f"Ref: {records[ref_idx]['name']}",
                     transform=ax_zoom.transAxes,
                     fontsize=6.5, ha="right", va="bottom",
                     fontfamily="serif", color=colors[ref_idx], style="italic")

    return fig
