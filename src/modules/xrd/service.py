
from .parser import XRDParser
from .engine import (
    sanitise_params, smooth, detect_peaks, fit_peaks,
    debye_scherrer, stokes_wilson, williamson_hall,
    dislocation_density, build_summary_table
)
import numpy as np
import pandas as pd


class XRDService:
    """
    Orchestrator for XRD analysis. 
    Ties together parsing, smoothing, fitting and crystallographic math.
    """
    
    @staticmethod
    def run_analysis(file_content, params: dict) -> dict:
        # 1. Parse
        df = XRDParser.parse(file_content)
        if len(df) < 10:
            raise ValueError("Insufficient data points after cleaning.")
            
        two_theta = df["two_theta"].values
        intensity = df["intensity"].values
        
        # 2. Params
        aa_window, sg_window, sg_poly, warns = sanitise_params(
            params['aa_window'], params['sg_window'], params['sg_poly'], len(intensity)
        )
        
        # 3. Smooth & Detect
        y_aa, y_fit = smooth(intensity, aa_window, sg_window, sg_poly)
        peak_idx = detect_peaks(y_fit, params['height_frac'], params['prom_frac'], params['peak_dist'])
        
        if len(peak_idx) == 0:
            raise ValueError("No peaks detected. Try lowering thresholds.")
            
        # 4. Fit
        fit_results = fit_peaks(
            two_theta, y_fit, peak_idx, 
            fwhm_min=params['fwhm_min'], 
            fwhm_max=params['fwhm_max']
        )
        
        if len(fit_results) == 0:
            raise ValueError("No peaks passed the FWHM sanity filter.")
            
        # 5. Physics
        D_DS_avg, D_list_DS = debye_scherrer(fit_results)
        eps_SW, eps_SW_std, eps_SW_list = stokes_wilson(fit_results)
        wh = williamson_hall(fit_results)
        
        primary_idx = np.argmax([r["height"] for r in fit_results])
        D_primary = D_list_DS[primary_idx]
        
        delta_primary = dislocation_density(D_primary)
        delta_WH = dislocation_density(wh["D_nm"])
        
        summary = build_summary_table(
            D_primary, wh["D_nm"], eps_SW, wh["eps"],
            delta_primary, delta_WH, delta_primary, 
            fit_results
        )
        
        # Peak detail table
        peak_rows = []
        for r, D, eps in zip(fit_results, D_list_DS, eps_SW_list):
            peak_rows.append({
                "2θ (°)":       round(r["two_theta_deg"], 4),
                "FWHM (°)":     round(r["fwhm_deg"],      4),
                "D_DS (nm)":    round(D, 3) if np.isfinite(D) else "—",
                "ε_SW (×10⁻³)": round(eps * 1e3, 4) if np.isfinite(eps) else "—",
                "Fit R²":       round(r["r_sq"], 4),
            })
            
        return {
            "two_theta":    two_theta,
            "intensity":    intensity,
            "y_aa":         y_aa,
            "fit_results":  fit_results,
            "wh":           wh,
            "D_DS":         D_DS_avg, 
            "D_primary":    D_primary,
            "eps_SW":       eps_SW,
            "eps_SW_std":   eps_SW_std,
            "eps_SW_list":  eps_SW_list,
            "D_list_DS":    D_list_DS,
            "delta_DS":     delta_primary,
            "delta_WH":     delta_WH,
            "summary":      summary,
            "peak_table":   pd.DataFrame(peak_rows),
            "warns":        warns,
            "aa_window":    aa_window,
        }

    @staticmethod
    def load_peak_shift_data(uploaded_file) -> dict | None:
        """Helper for peak shiftComparison."""
        try:
            from scipy.signal import find_peaks
            from scipy.optimize import curve_fit
            from scipy.special import voigt_profile
            
            df = XRDParser.parse(uploaded_file.getvalue())
            if len(df) < 10:
                return None
            
            x = df["two_theta"].values
            y = df["intensity"].values
            y_aa, y_fit = smooth(y, aa_window=20)
            
            i_range = y_fit.max() - y_fit.min()
            peaks, props = find_peaks(
                y_fit,
                height     = y_fit.min() + 0.05 * i_range,
                prominence = 0.05 * i_range,
                distance   = 20,
            )
            if len(peaks) == 0:
                return None
            
            dom_idx = peaks[np.argmax(props["prominences"])]
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
                
            return {"x": x, "y_aa": y_aa, "dom_idx": dom_idx, "center": center, "name": uploaded_file.name}
        except Exception:
            return None
