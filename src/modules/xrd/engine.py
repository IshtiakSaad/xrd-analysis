"""
xrd_engine.py
=============
Pure analysis functions shared by both analysis modes.
No file I/O, no Streamlit imports — fully testable standalone.

All functions include defensive error handling to prevent crashes
from unusual parameter combinations.
"""

import numpy as np
import pandas as pd
from scipy.signal   import savgol_filter, find_peaks
from scipy.optimize import curve_fit
from scipy.special  import voigt_profile
from scipy.stats    import linregress
import warnings
warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────
#  CONSTANTS
# ──────────────────────────────────────────────────────────────────
K_SCHERRER = 0.9
LAMBDA_NM  = 0.15406    # Cu Kα


# ──────────────────────────────────────────────────────────────────
#  PARAMETER VALIDATION
# ──────────────────────────────────────────────────────────────────
def sanitise_params(aa_window, sg_window, sg_poly, n_points):
    """
    Clamp / fix parameters to guarantee they are safe for the
    underlying scipy functions.  Returns corrected values + a list
    of warning strings (empty if nothing was changed).
    """
    warns = []

    # aa_window must be odd and >= 3, and <= n_points
    aa_window = int(max(3, min(aa_window, n_points)))
    if aa_window % 2 == 0:
        aa_window += 1

    # sg_window must be odd and >= 5
    sg_window = int(max(5, min(sg_window, n_points)))
    if sg_window % 2 == 0:
        sg_window += 1

    # sg_poly MUST be < sg_window  (scipy hard requirement)
    if sg_poly >= sg_window:
        new_poly = sg_window - 1
        warns.append(
            f"S-G polynomial order ({sg_poly}) must be < window "
            f"({sg_window}).  Clamped to {new_poly}."
        )
        sg_poly = new_poly

    # sg_poly at least 1
    sg_poly = int(max(1, sg_poly))

    return aa_window, sg_window, sg_poly, warns


# ──────────────────────────────────────────────────────────────────
#  SMOOTHING
# ──────────────────────────────────────────────────────────────────
def adjacent_average(y: np.ndarray, n: int) -> np.ndarray:
    """Uniform moving average — identical to Origin's Adjacent Averaging."""
    if n < 1:
        return y.copy()
    kernel = np.ones(n) / n
    padded = np.pad(y, (n // 2, n // 2), mode="reflect")
    return np.convolve(padded, kernel, mode="valid")[: len(y)]


def smooth(y: np.ndarray, aa_window: int = 20,
           sg_window: int = 15, sg_poly: int = 3):
    """
    Two-stage smoothing.
    Returns (y_display, y_fit):
      y_display — adjacent-averaged (what is plotted)
      y_fit     — additional S-G pass (used for peak detection / fitting)
    """
    y_aa = adjacent_average(y, aa_window)
    try:
        y_fit = savgol_filter(y_aa, window_length=sg_window, polyorder=sg_poly)
    except Exception:
        # Fallback: if S-G fails for any reason, just use AA result
        y_fit = y_aa.copy()
    return y_aa, y_fit


# ──────────────────────────────────────────────────────────────────
#  PEAK DETECTION
# ──────────────────────────────────────────────────────────────────
def detect_peaks(y_fit: np.ndarray,
                 height_frac: float = 0.05,
                 prominence_frac: float = 0.03,
                 distance: int = 20) -> np.ndarray:
    i_range = y_fit.max() - y_fit.min()
    if i_range <= 0:
        return np.array([], dtype=int)
    distance = max(1, distance)
    try:
        idx, _ = find_peaks(
            y_fit,
            height     = y_fit.min() + height_frac * i_range,
            prominence = prominence_frac * i_range,
            distance   = distance,
        )
    except Exception:
        idx = np.array([], dtype=int)
    return idx


# ──────────────────────────────────────────────────────────────────
#  VOIGT FITTING
# ──────────────────────────────────────────────────────────────────
def _voigt_model(x, amp, center, sigma, gamma, bg0, bg1):
    return amp * voigt_profile(x - center, sigma, gamma) + bg0 + bg1 * (x - center)


def _voigt_fwhm(sigma, gamma):
    fG = 2.0 * np.sqrt(2.0 * np.log(2.0)) * abs(sigma)
    fL = 2.0 * abs(gamma)
    return 0.5346 * fL + np.sqrt(0.2166 * fL ** 2 + fG ** 2)


def fit_peaks(two_theta: np.ndarray, y_fit: np.ndarray,
              peak_idx: np.ndarray,
              window_pts: int = 50,
              fwhm_min: float = 0.05,
              fwhm_max: float = 5.0) -> list[dict]:
    """
    Fit a Voigt profile to each detected peak.
    Returns list of dicts with: two_theta_deg, theta_rad, fwhm_deg,
    beta_rad, popt, x_win  — only peaks passing FWHM sanity filter.
    """
    results = []
    for idx in peak_idx:
        lo = max(0, idx - window_pts)
        hi = min(len(two_theta) - 1, idx + window_pts)
        xw = two_theta[lo:hi]
        yw = y_fit[lo:hi]
        if len(xw) < 8:
            continue
        try:
            popt, _ = curve_fit(
                _voigt_model, xw, yw,
                p0     = [yw.max() - yw.min(), two_theta[idx], 0.1, 0.1, yw[0], 0.0],
                bounds = ([0, xw.min(), 1e-5, 1e-5, -np.inf, -np.inf],
                          [np.inf, xw.max(), 5.0, 5.0, np.inf, np.inf]),
                maxfev = 10_000,
            )
            _, center, sigma, gamma, _, _ = popt
            fwhm_deg = _voigt_fwhm(sigma, gamma)
            if not (fwhm_min < fwhm_deg < fwhm_max):
                continue

            theta_rad = np.radians(center / 2.0)
            # Guard against theta exactly 0 or 90 degrees
            if abs(np.cos(theta_rad)) < 1e-12:
                continue

            # Calculate R-squared for fit quality
            y_model = _voigt_model(xw, *popt)
            ss_res = np.sum((yw - y_model) ** 2)
            ss_tot = np.sum((yw - np.mean(yw)) ** 2)
            r_sq = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

            results.append({
                "two_theta_deg": center,
                "theta_rad":     theta_rad,
                "fwhm_deg":      fwhm_deg,
                "beta_rad":      np.radians(fwhm_deg),
                "popt":          popt,
                "x_win":         xw,
                "r_sq":          r_sq,
                "height":        popt[0], # Amplitude from Voigt fit
            })
        except (RuntimeError, ValueError, TypeError):
            pass
    return results


# ──────────────────────────────────────────────────────────────────
#  CRYSTALLOGRAPHIC ANALYSIS
# ──────────────────────────────────────────────────────────────────
def debye_scherrer(fit_results: list[dict],
                   K: float = K_SCHERRER,
                   lam: float = LAMBDA_NM) -> tuple[float, list[float]]:
    """Returns (mean_D_nm, per_peak_D_list)."""
    D_list = []
    for r in fit_results:
        cos_t = np.cos(r["theta_rad"])
        if abs(cos_t) < 1e-12 or r["beta_rad"] < 1e-12:
            D_list.append(float("nan"))
        else:
            D_list.append((K * lam) / (r["beta_rad"] * cos_t))
    valid = [d for d in D_list if np.isfinite(d)]
    mean_D = float(np.mean(valid)) if valid else float("nan")
    return mean_D, D_list


def stokes_wilson(fit_results: list[dict]) -> tuple[float, float, list[float]]:
    """Returns (mean_eps, std_eps, per_peak_eps_list).  ε dimensionless."""
    eps_list = []
    for r in fit_results:
        tan_t = np.tan(r["theta_rad"])
        if abs(tan_t) < 1e-12:
            eps_list.append(float("nan"))
        else:
            eps_list.append(r["beta_rad"] / (4.0 * tan_t))
    valid = [e for e in eps_list if np.isfinite(e)]
    mean_eps = float(np.mean(valid)) if valid else float("nan")
    std_eps  = float(np.std(valid))  if len(valid) > 1 else 0.0
    return mean_eps, std_eps, eps_list


def williamson_hall(fit_results: list[dict],
                    K: float = K_SCHERRER,
                    lam: float = LAMBDA_NM):
    """
    Returns dict with keys:
      x, y, slope, intercept, r_sq, D_nm, eps, se_slope
    Handles edge cases: <2 points → returns NaN-based results.
    """
    x = np.array([4.0 * np.sin(r["theta_rad"]) for r in fit_results])
    y = np.array([r["beta_rad"] * np.cos(r["theta_rad"]) for r in fit_results])

    if len(x) < 2:
        # Can't do linear regression with <2 points
        intercept = y[0] if len(y) == 1 else 0.0
        D_nm = (K * lam) / intercept if intercept > 0 else float("nan")
        return {
            "x": x, "y": y,
            "slope": 0.0, "intercept": intercept,
            "r_sq": float("nan"),
            "D_nm": D_nm,
            "eps": 0.0,
            "se_slope": float("nan"),
        }

    try:
        slope, intercept, r_val, _, se = linregress(x, y)
    except Exception:
        return {
            "x": x, "y": y,
            "slope": 0.0, "intercept": 0.0,
            "r_sq": float("nan"),
            "D_nm": float("nan"),
            "eps": 0.0,
            "se_slope": float("nan"),
        }

    D_nm = (K * lam) / intercept if intercept > 0 else float("nan")
    return {
        "x": x, "y": y,
        "slope": slope, "intercept": intercept,
        "r_sq": r_val ** 2,
        "D_nm": D_nm,
        "eps": float(slope),
        "se_slope": se,
    }


def dislocation_density(D_nm: float) -> float:
    """δ = 1/D²  in units of ×10⁻³ nm⁻²."""
    if D_nm > 0 and np.isfinite(D_nm):
        return (1.0 / D_nm ** 2) * 1e3
    return float("nan")


def build_summary_table(D_val, D_WH, eps_SW_mean, eps_WH,
                        delta_val, delta_WH, delta_SW,
                        fit_results: list[dict]) -> pd.DataFrame:
    def _f(v, n=4):
        if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
            return "—"
        return str(round(float(v), n))

    # Average R-squared for trust scoring
    avg_r2 = np.mean([r["r_sq"] for r in fit_results]) if fit_results else 0.0

    return pd.DataFrame([
        {
            "Method":                              "Debye–Scherrer (Primary)",
            "Crystallite size D (nm)":             _f(D_val,  3),
            "Micro-strain ε (×10⁻³)":             "—",
            "Dislocation density δ (×10⁻³ nm⁻²)": _f(delta_val, 6),
            "Fit Quality (Avg R²)":                _f(avg_r2, 4),
        },
        {
            "Method":                              "Stokes–Wilson (Average)",
            "Crystallite size D (nm)":             _f(D_val,  3),
            "Micro-strain ε (×10⁻³)":             _f(eps_SW_mean * 1e3, 4),
            "Dislocation density δ (×10⁻³ nm⁻²)": _f(delta_SW, 6),
            "Fit Quality (Avg R²)":                _f(avg_r2, 4),
        },
        {
            "Method":                              "Williamson–Hall (Regression)",
            "Crystallite size D (nm)":             _f(D_WH,  3),
            "Micro-strain ε (×10⁻³)":             _f(eps_WH  * 1e3, 4),
            "Dislocation density δ (×10⁻³ nm⁻²)": _f(delta_WH, 6),
            "Fit Quality (Avg R²)":                _f(avg_r2, 4),
        },
    ])
