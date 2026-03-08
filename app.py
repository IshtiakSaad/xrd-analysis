
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.shared.styles import inject_custom_css
from src.modules.xrd.service import XRDService
from src.modules.xrd.plots import (
    fig_xrd_pattern, fig_williamson_hall, fig_stokes_wilson, fig_combined, fig_peak_shift
)
import io
import zipfile

# ── PAGE CONFIG ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Scientific Analysis Platform",
    page_icon="🔬",
    layout="wide"
)

inject_custom_css()

# ── SIDEBAR ROUTING ──────────────────────────────────────────────
with st.sidebar:
    st.title("🔬 LabPro Analysis")
    st.caption("v2.0 • Professional Edition")
    
    analysis_type = st.radio(
        "Select Technique",
        ["XRD (X-Ray Diffraction)", "FTIR (Coming Soon)", "UV-Vis (Coming Soon)"]
    )
    
    if "XRD" in analysis_type:
        st.divider()
        mode = st.radio("Mode", ["Single Sample", "Peak Shift"])
        
        st.subheader("Instrument Settings")
        lam = st.number_input("λ (nm)", value=0.15406, format="%.5f")
        k_factor = st.number_input("K factor", value=0.9, format="%.2f")
        
        with st.expander("⚙️ Processing Params"):
            aa_win = st.slider("Smoothing Window", 5, 51, 20, step=2)
            h_frac = st.slider("Height threshold %", 1, 20, 5) / 100
            p_frac = st.slider("Prominence %", 1, 20, 3) / 100
            
    st.info("💡 Support development")
    st.button("☕ Buy me a coffee")


# ── ANALYSIS LOGIC (XRD) ──────────────────────────────────────────
if "XRD" in analysis_type:
    st.markdown(f"### XRD Crystallographic Analysis — {mode}")
    st.caption("Upload a file with two columns: 2θ (degrees) and Intensity. The file may have any header — only the first two columns are used.")

    if mode == "Single Sample":
        uploaded = st.file_uploader("Drop your file here", type=["csv", "txt", "raw"])
        
        if uploaded:
            params = {
                'aa_window': aa_win, 'sg_window': 15, 'sg_poly': 3,
                'height_frac': h_frac, 'prom_frac': p_frac,
                'peak_dist': 20, 'fwhm_min': 0.05, 'fwhm_max': 5.0
            }
            
            try:
                res = XRDService.run_analysis(uploaded.getvalue(), params)
                
                # Success banner
                st.success(f"✓ {len(res['fit_results'])} peaks detected and fitted successfully.")
                
                # ── RESULTS AT A GLANCE ─────────────────────────────────────
                st.subheader("Results at a Glance")
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Peaks Fitted", len(res['fit_results']))
                m2.metric("D (Debye-Scherrer)", f"{res['D_primary']:.2f} nm")
                m3.metric("D (Williamson-Hall)", f"{res['wh']['D_nm']:.2f} nm")
                m4.metric("ε (Stokes-Wilson)", f"{res['eps_SW']*1e3:.3f} ×10⁻³")
                m5.metric("ε (Williamson-Hall)", f"{res['wh']['eps']*1e3:.3f} ×10⁻³")
                
                # ── SUMMARY TABLE ──────────────────────────────────────────
                st.subheader("Summary Table")
                st.dataframe(res['summary'], use_container_width=True, hide_index=True)
                st.caption("*Stokes-Wilson is an upper bound (all broadening attributed to strain). Williamson-Hall separates size and strain contributions.*")
                
                with st.expander("📖 Mathematical Explainability & Bias Report"):
                    st.write("""
                    **Physics Postulates Applied:**
                    1. **Scherrer Legacy**: We use $K=0.9$ (spherical assumption). 
                    2. **Voigt Deconvolution**: Peaks are fitted using a Voigt profile to account for both Gaussian (instrumental/strain) and Lorentzian (size) lifetime broadening.
                    3. **Stokes-Wilson Approximation**: Assumes the entirety of line broadening $\beta$ is due to lattice micro-strain.
                    4. **Williamson-Hall Regression**: Decouples size ($D$) and strain ($\epsilon$) using the linear relationship $\beta \cos \theta = \epsilon (4 \sin \theta) + \frac{K\lambda}{D}$.
                    
                    **Potential Biases:**
                    - *Instrumental Broadening*: This app currently assumes a high-resolution diffractometer. If your instrument has significant intrinsic broadening, crystallite sizes $>100$nm may be underestimated.
                    - *Texture/Preferred Orientation*: High-intensity peaks are prioritized for 'Primary' calculations.
                    """)

                # ── FIGURES ────────────────────────────────────────────────
                st.subheader("Figures")
                t1, t2, t3, t4 = st.tabs(["XRD Pattern", "Williamson-Hall", "Stokes-Wilson", "Combined panel"])
                with t1:
                    st.pyplot(fig_xrd_pattern(res['two_theta'], res['intensity'], res['y_aa'], res['fit_results'], res['aa_window']))
                    st.markdown('<p class="fig-caption">Fig 1. — Experimental vs. fitted XRD pattern with detected peak indices.</p>', unsafe_allow_html=True)
                with t2:
                    st.pyplot(fig_williamson_hall(res['fit_results'], res['wh']))
                    st.markdown('<p class="fig-caption">Fig 2. — Williamson-Hall plot for size-strain separation.</p>', unsafe_allow_html=True)
                with t3:
                    st.pyplot(fig_stokes_wilson(res['fit_results'], res['eps_SW_list'], res['eps_SW']))
                    st.markdown('<p class="fig-caption">Fig 3. — Quantitative micro-strain per crystallographic plane.</p>', unsafe_allow_html=True)
                with t4:
                    st.pyplot(fig_combined(res['two_theta'], res['intensity'], res['y_aa'], res['fit_results'], 
                                          res['wh'], res['eps_SW_list'], res['eps_SW']))
                    st.markdown('<p class="fig-caption">Fig 4. — Combined three-panel figure (full journal-page width).</p>', unsafe_allow_html=True)

                # ── PEAK DETAIL TABLE ──────────────────────────────────────
                with st.expander("▽ Per-peak detail table", expanded=False):
                    st.dataframe(res['peak_table'], use_container_width=True, hide_index=True)
                
                # ── DOWNLOAD SECTION ───────────────────────────────────────
                st.subheader("Download")
                st.caption("All figures as PNG (300 DPI) and PDF (vector) plus the summary tables as CSV, bundled in one zip.")
                
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "x") as zf:
                    # Save Figures
                    for fmt in ["png", "pdf"]:
                        f1 = fig_xrd_pattern(res['two_theta'], res['intensity'], res['y_aa'], res['fit_results'], res['aa_window'])
                        img_buf = io.BytesIO(); f1.savefig(img_buf, format=fmt, dpi=300); zf.writestr(f"xrd_pattern.{fmt}", img_buf.getvalue())
                        f2 = fig_williamson_hall(res['fit_results'], res['wh'])
                        img_buf = io.BytesIO(); f2.savefig(img_buf, format=fmt, dpi=300); zf.writestr(f"williamson_hall.{fmt}", img_buf.getvalue())
                        f3 = fig_combined(res['two_theta'], res['intensity'], res['y_aa'], res['fit_results'], res['wh'], res['eps_SW_list'], res['eps_SW'])
                        img_buf = io.BytesIO(); f3.savefig(img_buf, format=fmt, dpi=300); zf.writestr(f"combined_panel.{fmt}", img_buf.getvalue())
                    
                    # Save CSVs
                    zf.writestr("summary_results.csv", res['summary'].to_csv(index=False))
                    zf.writestr("peak_details.csv", res['peak_table'].to_csv(index=False))
                
                st.download_button(
                    label="⬇ Download all figures + tables (.zip)",
                    data=buf.getvalue(),
                    file_name=f"XRD_Analysis_{uploaded.name.split('.')[0]}.zip",
                    mime="application/zip"
                )
                    
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                st.info("Tip: Ensure your file has numeric data and the correct 2-column format.")
                
    else:
        # ── PEAK SHIFT MODE ──────────────────────────────────────────
        st.subheader("Peak Shift / Lattice Expansion Analysis")
        st.info("Upload multiple samples to compare peak migration (e.g., thermal expansion or doping).")
        multi_files = st.file_uploader("Upload 2-10 files", type=["csv","txt","raw"], accept_multiple_files=True)
        
        if multi_files and len(multi_files) >= 2:
            records = []
            for f in multi_files:
                rec = XRDService.load_peak_shift_data(f)
                if rec: records.append(rec)
            
            if len(records) >= 2:
                ref_name = st.selectbox("Select Reference Sample (Control)", [r["name"] for r in records])
                ref_idx = next(i for i, r in enumerate(records) if r["name"] == ref_name)
                
                st.pyplot(fig_peak_shift(records, ref_idx=ref_idx))
                st.markdown('<p class="fig-caption">Fig 5. — Multi-sample peak shift comparison (Normalized & Stacked).</p>', unsafe_allow_html=True)
            else:
                st.warning("Could not extract dominant peaks from at least 2 files.")
        elif multi_files:
            st.info("Upload at least 2 files to enable comparison.")
else:
    st.warning("This module is currently in development. Please select XRD to continue.")
