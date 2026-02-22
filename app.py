"""
XRD Analysis Web App
====================
Run with:  streamlit run app.py
"""

import io
import traceback
import zipfile
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal  import savgol_filter, find_peaks
from scipy.optimize import curve_fit
from scipy.special  import voigt_profile

from xrd_engine import (
    run_single_sample, adjacent_average, smooth,
    detect_peaks, fit_peaks, sanitise_params,
    debye_scherrer, stokes_wilson, williamson_hall,
    dislocation_density,
)
from xrd_plots import (
    fig_xrd_pattern, fig_williamson_hall,
    fig_stokes_wilson, fig_combined, fig_peak_shift,
    PEAK_COLORS,
)


# ══════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="XRD Analysis Tool",
    page_icon="⚛",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS — sleek dark academic look ────────────────────────────
def inject_custom_css():
    # Force sleek Dark theme for the UI
    bg_main   = "#0E1117"
    bg_sec    = "#1E1E1E"
    text_main = "#E0E0E0"
    text_muted= "#999999"
    accent    = "#FF4B4B"
    border    = "#333333"
    card_shadow = "0 4px 20px rgba(0,0,0,0.3)"
    
    css = f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600&family=EB+Garamond:ital,wght@0,400;0,500;1,400&family=JetBrains+Mono:wght@400;500&display=swap');

      /* Global resets */
      html, body, [data-testid="stAppViewContainer"] {{
        font-family: 'Outfit', sans-serif;
        background-color: {bg_main};
        color: {text_main};
      }}
      
      /* Typography */
      h1, h2, h3 {{ font-family: 'Outfit', sans-serif; letter-spacing: -0.02em; }}
      h1 {{ font-size: 2.8rem !important; font-weight: 600; margin-bottom: 2rem !important; color: {text_main}; }}
      h2 {{ font-size: 1.6rem !important; font-weight: 500; border-bottom: 2px solid {accent}33; padding-bottom: 0.5rem; margin-top: 2.5rem !important; color: {text_main}; }}
      
      /* Academic text styles */
      p, li {{ font-size: 1.1rem; line-height: 1.6; }}

      /* Sidebar Refinement */
      section[data-testid="stSidebar"] {{ 
        background-color: {bg_sec} !important; 
        border-right: 1px solid {border}; 
        box-shadow: 10px 0 30px rgba(0,0,0,0.02);
      }}
      
      /* Card-like components */
      div[data-testid="metric-container"] {{
        background-color: {bg_sec} !important;
        border: 1px solid {border};
        border-radius: 16px;
        padding: 1.2rem;
        box-shadow: {card_shadow};
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
      }}
      div[data-testid="metric-container"]:hover {{
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.08);
        border-color: {accent}55;
      }}
      
      div[data-testid="stMetricValue"] {{
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        color: {accent} !important;
      }}

      /* File Uploader styling */
      div[data-testid="stFileUploader"] {{
        border: 2px dashed {border};
        background-color: {bg_sec};
        border-radius: 16px;
        padding: 2rem;
        transition: all 0.3s ease;
      }}
      div[data-testid="stFileUploader"]:hover {{
        border-color: {accent};
        background-color: {accent}05;
      }}

      /* Tab Styling */
      button[data-baseweb="tab"] {{
        font-family: 'Outfit', sans-serif;
        font-size: 1rem;
        font-weight: 500;
        color: {text_muted};
        transition: all 0.2s ease;
      }}
      button[data-baseweb="tab"][aria-selected="true"] {{
        color: {accent} !important;
        border-bottom-color: {accent} !important;
      }}
      
      /* Expander Styling */
      div[data-testid="stExpander"] {{
        border: 1px solid {border};
        border-radius: 12px;
        background-color: {bg_sec} !important;
        margin-bottom: 1rem;
      }}

      /* Button Styling */
      .stButton > button {{
        border-radius: 8px;
        font-weight: 500;
        padding: 0.5rem 1rem;
        transition: all 0.2s ease;
      }}
      
      /* Figure Captions */
      .fig-caption {{
        font-family: 'EB Garamond', serif;
        font-size: 1rem;
        color: {text_muted};
        text-align: center;
        margin: 1rem 0 3rem 0;
        font-style: italic;
      }}

      /* Hide scientific notation in metrics if any */
      small {{ color: {text_muted}; }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)





# ══════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════
def fig_to_bytes(fig: plt.Figure, fmt: str = "png", dpi: int = 300) -> bytes:
    buf = io.BytesIO()
    try:
        fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight")
    except Exception:
        # Fallback: lower DPI
        fig.savefig(buf, format=fmt, dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


def make_zip(files: dict[str, bytes]) -> bytes:
    """files = {filename: bytes}"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return buf.getvalue()


def caption(text: str):
    st.markdown(f'<p class="fig-caption">{text}</p>', unsafe_allow_html=True)


def safe_format(value, fmt=".2f", fallback="—"):
    """Format a numeric value, returning fallback for NaN/Inf."""
    try:
        if np.isfinite(value):
            return f"{value:{fmt}}"
    except (TypeError, ValueError):
        pass
    return fallback


def load_peak_shift_csv(uploaded) -> dict | None:
    """Load one CSV for peak-shift mode; return engine record dict."""
    try:
        raw = uploaded.read()
        df  = pd.read_csv(io.BytesIO(raw), header=0)
        if df.shape[1] < 2:
            return None
        df.columns = list(df.columns[:2])
        df.columns = ["two_theta", "intensity"]
        df = df.dropna().astype(float).sort_values("two_theta").reset_index(drop=True)
        if len(df) < 10:
            return None
        x = df["two_theta"].values
        y = df["intensity"].values
        y_aa, y_fit = smooth(y, aa_window=20)
        i_range = y_fit.max() - y_fit.min()
        if i_range <= 0:
            return None
        peaks, props = find_peaks(
            y_fit,
            height     = y_fit.min() + 0.05 * i_range,
            prominence = 0.05 * i_range,
            distance   = 20,
        )
        if len(peaks) == 0:
            return None
        dom_idx = peaks[np.argmax(props["prominences"])]
        # Voigt fit for precise centre
        lo = max(0, dom_idx - 60); hi = min(len(x) - 1, dom_idx + 60)
        xw = x[lo:hi]; yw = y_fit[lo:hi]
        try:
            popt, _ = curve_fit(
                lambda x, a, c, s, g, b0, b1:
                    a * voigt_profile(x - c, s, g) + b0 + b1 * (x - c),
                xw, yw,
                p0=[yw.max()-yw.min(), x[dom_idx], 0.1, 0.1, yw[0], 0.],
                bounds=([0,xw.min(),1e-5,1e-5,-np.inf,-np.inf],
                        [np.inf,xw.max(),5,5,np.inf,np.inf]),
                maxfev=10_000)
            center = popt[1]
        except Exception:
            center = x[dom_idx]
        return {"x": x, "y_aa": y_aa, "dom_idx": dom_idx, "center": center}
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════
#  SIDEBAR  (simplified — advanced options hidden by default)
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚛ XRD Analysis")
    st.markdown("#### Developed by Ishtiak Saad at RUET")
    st.markdown("---")

    mode = st.radio(
        "Analysis mode",
        ["Single Sample", "Peak Shift (multi-sample)"],
        help="Single sample: full crystallographic analysis.\n"
             "Peak shift: overlay & compare dominant peaks across samples.",
    )

    st.markdown("---")
    st.markdown("### Instrument")
    lam = st.number_input("Wavelength λ (nm)", value=0.15406, format="%.5f",
                          help="Cu Kα = 0.15406 nm",
                          min_value=0.01, max_value=1.0)
    K   = st.number_input("Scherrer constant K", value=0.9, format="%.2f",
                          min_value=0.1, max_value=2.0)

    inject_custom_css()

    # ── Advanced parameters (collapsed) ─────────────────────────
    with st.expander("⚙ Advanced parameters", expanded=False):
        st.caption("⚠ Only adjust these if the defaults don't work for your data.")

        st.markdown("**Smoothing**")
        aa_window = st.slider("Adjacent-avg window", 5, 51, 20, step=2,
                              help="Matches Origin's 'Adjacent Averaging' method",
                              key="aa_window")
        sg_window = st.slider("S-G window (fitting)", 5, 51, 15, step=2,
                              key="sg_window")
        sg_poly   = st.slider("S-G poly order", 1, 6, 3,
                              key="sg_poly")

        # Live validation feedback
        if sg_poly >= sg_window:
            st.warning(
                f"⚠ Poly order ({sg_poly}) must be less than "
                f"S-G window ({sg_window}). It will be auto-clamped.",
                icon="⚠️"
            )

        st.markdown("**Peak Detection**")
        height_frac = st.slider("Min. height (% of range)", 1, 20, 5,
                                key="height_frac") / 100
        prom_frac   = st.slider("Min. prominence (%)", 1, 20, 3,
                                key="prom_frac") / 100
        peak_dist   = st.slider("Min. peak separation (pts)", 5, 100, 20,
                                key="peak_dist")

        st.markdown("**Peak Fitting**")
        fwhm_min = st.number_input("FWHM min (°)", value=0.05, format="%.3f",
                                   min_value=0.001, max_value=2.0,
                                   key="fwhm_min")
        fwhm_max = st.number_input("FWHM max (°)", value=5.0, format="%.2f",
                                   min_value=0.1, max_value=20.0,
                                   key="fwhm_max")

        # Cross-validation
        if fwhm_min >= fwhm_max:
            st.error("FWHM min must be less than FWHM max.")

    st.markdown("---")
    st.markdown(
        "<small>Publication figures: 300 DPI, Times New Roman, "
        "full-box axes, inward ticks.<br>"
        "PNG + PDF provided for every figure.</small>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════
#  MODE A — SINGLE SAMPLE
# ══════════════════════════════════════════════════════════════════
if mode == "Single Sample":

    st.title("XRD Crystallographic Analysis")
    st.markdown(
        "Upload a CSV with two columns: **2θ (degrees)** and **Intensity**. "
        "The file may have any header — only the first two columns are used."
    )

    uploaded = st.file_uploader(
        "Drop your CSV here", type=["csv", "txt"],
        help="Format: first column = 2θ, second column = intensity. "
             "Header row optional.",
    )

    if uploaded is None:
        st.info("👆 Upload a CSV file to begin.")
        st.stop()

    # ── Run analysis ────────────────────────────────────────────
    with st.spinner("Running analysis…"):
        try:
            result = run_single_sample(
                uploaded.read(),
                aa_window=aa_window, sg_window=sg_window, sg_poly=sg_poly,
                height_frac=height_frac, prom_frac=prom_frac,
                peak_dist=peak_dist, fwhm_min=fwhm_min, fwhm_max=fwhm_max,
            )
        except ValueError as e:
            st.error(f"⚠ Analysis failed: {e}")
            st.info(
                "💡 **Tip:** Try opening the *Advanced parameters* panel in "
                "the sidebar and adjusting the thresholds. Common fixes:\n"
                "- Lower the **Min. height** or **Prominence** sliders\n"
                "- Widen the **FWHM** range\n"
                "- Reduce **Peak separation**"
            )
            st.stop()
        except Exception as e:
            st.error(
                f"⚠ An unexpected error occurred: {e}\n\n"
                "Please check that your CSV is properly formatted "
                "(two numeric columns: 2θ and Intensity)."
            )
            st.stop()

    # Show parameter auto-correction warnings if any
    for w in result.get("param_warns", []):
        st.warning(f"⚙ {w}")

    two_theta   = result["two_theta"]
    intensity   = result["intensity"]
    y_aa        = result["y_aa"]
    fit_results = result["fit_results"]
    wh          = result["wh"]
    D_DS        = result["D_DS"]
    eps_SW      = result["eps_SW"]
    eps_SW_std  = result["eps_SW_std"]
    eps_SW_list = result["eps_SW_list"]
    D_list_DS   = result["D_list_DS"]
    delta_DS    = result["delta_DS"]
    delta_WH    = result["delta_WH"]
    summary     = result["summary"]
    peak_table  = result["peak_table"]

    n_peaks = len(fit_results)
    st.success(f"✓ {n_peaks} peak{'s' if n_peaks != 1 else ''} detected and fitted successfully.")

    # ── Key metrics ────────────────────────────────────────────
    st.markdown("## Results at a Glance")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Peaks fitted",        f"{n_peaks}")
    c2.metric("D  (Debye–Scherrer)", safe_format(D_DS, ".2f") + " nm")
    c3.metric("D  (Williamson–Hall)", safe_format(wh['D_nm'], ".2f") + " nm")
    c4.metric("ε  (Stokes–Wilson)",  safe_format(eps_SW*1e3, ".3f") + " ×10⁻³")
    c5.metric("ε  (Williamson–Hall)", safe_format(wh['eps']*1e3, ".3f") + " ×10⁻³")

    # Warn if W-H had insufficient data
    if n_peaks < 2:
        st.info(
            "ℹ Only 1 peak was fitted — Williamson–Hall analysis requires "
            "≥ 2 peaks for a linear regression. W-H values are approximations."
        )

    # ── Summary table ───────────────────────────────────────────
    st.markdown("## Summary Table")
    st.dataframe(summary, use_container_width=True, hide_index=True)
    st.markdown(
        "*Stokes–Wilson ε is an upper bound (all broadening attributed to strain). "
        "Williamson–Hall separates size and strain contributions.*"
    )

    # ── Figures ─────────────────────────────────────────────────
    st.markdown("## Figures")

    # Only show W-H and combined tabs if we have enough data
    if n_peaks >= 2:
        tab1, tab2, tab3, tab4 = st.tabs(
            ["XRD Pattern", "Williamson–Hall", "Stokes–Wilson", "Combined panel"]
        )
    else:
        tab1, tab3 = st.tabs(["XRD Pattern", "Stokes–Wilson"])
        tab2 = tab4 = None

    with tab1:
        try:
            # Figure uses white background for publication
            f1 = fig_xrd_pattern(two_theta, intensity, y_aa, fit_results, aa_window, is_dark=False)
            st.pyplot(f1, use_container_width=False)
            caption("Fig. 1 — Smoothed XRD pattern with fitted peak positions labelled.")
            plt.close(f1)
        except Exception as e:
            st.error(f"Could not render XRD pattern: {e}")

    if tab2 is not None:
        with tab2:
            try:
                f2 = fig_williamson_hall(fit_results, wh, is_dark=False)
                st.pyplot(f2, use_container_width=False)
                caption("Fig. 2 — Williamson–Hall plot. "
                        "Slope → micro-strain ε; intercept → Kλ/D.")
                plt.close(f2)
            except Exception as e:
                st.error(f"Could not render Williamson–Hall plot: {e}")

    with tab3:
        try:
            f3 = fig_stokes_wilson(fit_results, eps_SW_list, eps_SW, is_dark=False)
            st.pyplot(f3, use_container_width=False)
            caption("Fig. 3 — Stokes–Wilson micro-strain per peak (upper bound).")
            plt.close(f3)
        except Exception as e:
            st.error(f"Could not render Stokes–Wilson plot: {e}")

    if tab4 is not None:
        with tab4:
            try:
                f4 = fig_combined(two_theta, intensity, y_aa, fit_results,
                                  wh, eps_SW_list, eps_SW, aa_window, is_dark=False)
                st.pyplot(f4, use_container_width=True)
                caption("Fig. 4 — Combined three-panel figure (full journal-page width).")
                plt.close(f4)
            except Exception as e:
                st.error(f"Could not render combined panel: {e}")

    # ── Per-peak table ──────────────────────────────────────────
    with st.expander("Per-peak detail table"):
        st.dataframe(peak_table, use_container_width=True, hide_index=True)

    # ── Download zip ────────────────────────────────────────────
    st.markdown("## Download")
    st.markdown("All figures as **PNG (300 DPI)** and **PDF (vector)** "
                "plus the summary tables as CSV, bundled in one zip.")

    try:
        # Re-generate figures for download (ALWAYS light mode for publications)
        download_files = {}

        f1 = fig_xrd_pattern(two_theta, intensity, y_aa, fit_results, aa_window, is_dark=False)
        download_files["fig1_xrd_pattern.png"] = fig_to_bytes(f1, "png")
        download_files["fig1_xrd_pattern.pdf"] = fig_to_bytes(f1, "pdf")
        plt.close(f1)

        if n_peaks >= 2:
            f2 = fig_williamson_hall(fit_results, wh, is_dark=False)
            download_files["fig2_williamson_hall.png"] = fig_to_bytes(f2, "png")
            download_files["fig2_williamson_hall.pdf"] = fig_to_bytes(f2, "pdf")
            plt.close(f2)

        f3 = fig_stokes_wilson(fit_results, eps_SW_list, eps_SW, is_dark=False)
        download_files["fig3_stokes_wilson.png"] = fig_to_bytes(f3, "png")
        download_files["fig3_stokes_wilson.pdf"] = fig_to_bytes(f3, "pdf")
        plt.close(f3)

        if n_peaks >= 2:
            f4 = fig_combined(two_theta, intensity, y_aa, fit_results,
                              wh, eps_SW_list, eps_SW, aa_window, is_dark=False)
            download_files["fig4_combined.png"] = fig_to_bytes(f4, "png")
            download_files["fig4_combined.pdf"] = fig_to_bytes(f4, "pdf")
            plt.close(f4)

        download_files["summary_table.csv"] = summary.to_csv(index=False).encode()
        download_files["per_peak_table.csv"] = peak_table.to_csv(index=False).encode()

        zip_bytes = make_zip(download_files)

        st.download_button(
            "⬇ Download all figures + tables (.zip)",
            data=zip_bytes,
            file_name="xrd_analysis_results.zip",
            mime="application/zip",
        )
    except Exception as e:
        st.error(f"Could not generate download package: {e}")


# ══════════════════════════════════════════════════════════════════
#  MODE B — PEAK SHIFT (multi-sample)
# ══════════════════════════════════════════════════════════════════
else:
    st.title("XRD Peak Shift Comparison")
    st.markdown(
        "Upload **2–8 CSV files** (same format as single-sample mode). "
        "The app overlays all patterns and zooms in on the dominant peak "
        "of each sample to show Δ2θ shifts relative to your chosen reference."
    )

    uploaded_files = st.file_uploader(
        "Drop CSVs here (2–8 files)",
        type=["csv", "txt"],
        accept_multiple_files=True,
        help="Each file: first column = 2θ, second = intensity.",
    )

    if not uploaded_files:
        st.info("👆 Upload at least 2 CSV files to begin.")
        st.stop()

    if len(uploaded_files) < 2:
        st.warning("Please upload at least 2 files for comparison.")
        st.stop()

    if len(uploaded_files) > 8:
        st.error("Maximum 8 files supported.")
        st.stop()

    # ── Sample names ────────────────────────────────────────────
    st.markdown("### Sample labels")
    name_cols = st.columns(len(uploaded_files))
    sample_names = []
    for i, (col, f) in enumerate(zip(name_cols, uploaded_files)):
        default = f.name.replace(".csv","").replace(".txt","")
        name = col.text_input(f"File {i+1}", value=default, key=f"name_{i}")
        sample_names.append(name)

    ref_idx = st.selectbox(
        "Reference sample (Δ2θ = 0)",
        options=list(range(len(uploaded_files))),
        format_func=lambda i: sample_names[i],
    )

    zoom_margin = st.slider("Zoom window margin (°)", 0.2, 5.0, 1.2, 0.1)
    stack_frac  = st.slider("Stack offset (fraction of max intensity)",
                             0.0, 0.5, 0.15, 0.01)

    # ── Load all files ──────────────────────────────────────────
    with st.spinner("Processing files…"):
        records = []
        errors  = []
        for i, (f, name) in enumerate(zip(uploaded_files, sample_names)):
            rec = load_peak_shift_csv(f)
            if rec is None:
                errors.append(f"No peaks found in **{name}**.")
            else:
                rec["name"] = name
                records.append(rec)

    if errors:
        for e in errors:
            st.error(e)
        if len(records) < 2:
            st.error("Need at least 2 valid files for comparison.")
            st.stop()

    # Guard: ref_idx may be out of range if some files failed
    if ref_idx >= len(records):
        st.warning(
            f"Reference sample index ({ref_idx}) is out of range. "
            f"Using first sample as reference."
        )
        ref_idx = 0

    ref_center = records[ref_idx]["center"]

    # ── Peak position table ─────────────────────────────────────
    st.markdown("## Peak Positions")
    rows = []
    for i, r in enumerate(records):
        shift = r["center"] - ref_center
        rows.append({
            "Sample":           r["name"],
            "2θ peak (°)":      round(r["center"], 4),
            "Δ2θ from ref (°)": f"{shift:+.4f}",
            "Reference?":       "✓" if i == ref_idx else "",
        })
    shift_df = pd.DataFrame(rows)
    st.dataframe(shift_df, use_container_width=True, hide_index=True)

    # ── Figure ──────────────────────────────────────────────────
    st.markdown("## Peak Shift Figure")
    try:
        with st.spinner("Generating figure…"):
            # Force white background for publication plot
            f_ps = fig_peak_shift(records, ref_idx=ref_idx,
                                  zoom_margin=zoom_margin,
                                  stack_frac=stack_frac, is_dark=False)

        st.pyplot(f_ps, use_container_width=True)
        st.markdown(
            '<p class="fig-caption">'
            'Fig. — (a) Full-range stacked XRD patterns. '
            '(b) Zoomed view of dominant peak; patterns normalised 0–1 within window; '
            'Δ2θ shifts relative to reference shown.'
            '</p>',
            unsafe_allow_html=True,
        )
        plt.close(f_ps)
    except Exception as e:
        st.error(f"Could not render peak shift figure: {e}")

    # ── Download ─────────────────────────────────────────────────
    st.markdown("## Download")
    try:
        # Re-generate for download in light mode
        f_ps_dl = fig_peak_shift(records, ref_idx=ref_idx,
                                 zoom_margin=zoom_margin,
                                 stack_frac=stack_frac, is_dark=False)
        zip_bytes = make_zip({
            "fig_peak_shift.png":     fig_to_bytes(f_ps_dl, "png"),
            "fig_peak_shift.pdf":     fig_to_bytes(f_ps_dl, "pdf"),
            "peak_shift_table.csv":   shift_df.to_csv(index=False).encode(),
        })
        plt.close(f_ps_dl)

        st.download_button(
            "⬇ Download figure + table (.zip)",
            data=zip_bytes,
            file_name="xrd_peak_shift_results.zip",
            mime="application/zip",
        )
    except Exception as e:
        st.error(f"Could not generate download package: {e}")
