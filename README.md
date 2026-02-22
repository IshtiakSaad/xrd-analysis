# XRD Analysis Web App

A Streamlit web application for publication-quality XRD analysis.

## Features

**Single Sample mode**
- Adjacent-averaging smoothing (matches Origin)
- Automatic peak detection + Voigt profile fitting
- Debye–Scherrer crystallite size
- Stokes–Wilson micro-strain (per peak)
- Williamson–Hall crystallite size + micro-strain
- Dislocation density
- Download: PNG + PDF figures (300 DPI, Times New Roman, journal-ready) + CSV tables

**Peak Shift mode**
- Upload 2–8 CSV files simultaneously
- Custom sample labels + reference selection
- Full-range stacked overlay + zoomed normalised peak window
- Δ2θ shifts annotated on figure
- Download: figure + shift table

## CSV format

Two columns, any header (or no header):

```
2Theta, Intensity
10.00,  356
10.02,  406
...
```

---

## Quick start (local)

```bash
# 1. Clone or unzip this folder
cd xrd_app

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
streamlit run app.py
```

The app opens at **http://localhost:8501** in your browser.

---

## Deploy to Streamlit Cloud (free, public URL)

1. Push this folder to a **public GitHub repository**
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **New app** → select your repo → set **Main file path** to `app.py`
4. Click **Deploy** — done. You get a public URL to share with anyone.

> Streamlit Cloud installs `requirements.txt` automatically.

---

## Deploy with Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t xrd-app .
docker run -p 8501:8501 xrd-app
```

---

## File structure

```
xrd_app/
├── app.py           ← Streamlit UI (run this)
├── xrd_engine.py    ← Pure analysis functions (numpy/scipy)
├── xrd_plots.py     ← Publication figure generation (matplotlib)
├── requirements.txt
└── README.md
```

## Parameters (sidebar)

| Parameter | Default | Description |
|---|---|---|
| λ (nm) | 0.15406 | X-ray wavelength (Cu Kα) |
| K | 0.9 | Scherrer constant |
| Adjacent-avg window | 20 | Origin-equivalent smoothing |
| S-G window | 15 | Savitzky-Golay for peak fitting |
| Min. height | 5% | Peak detection threshold |
| Min. prominence | 3% | Peak detection threshold |
| FWHM min/max | 0.05/5.0° | Sanity filter for Voigt fits |
