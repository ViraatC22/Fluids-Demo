# Fluids Flow Lab (Streamlit)

Interactive Streamlit app for visualizing **continuity** and **Bernoulli’s principle** in an idealized pipe flow. The simulation shows tracer “water” particles accelerating through a constriction and provides an in-pipe hover/touch probe for local speed and pressure.

## Features

- Real-time 2D pipe flow visualization (self-contained HTML canvas)
- Continuity-driven velocity changes with pipe diameter
- Bernoulli + elevation effects in the local pressure estimate
- Data logging table with professional headers (units included) + CSV export
- Explanation tab suitable for lab reports / classroom use

## Run Locally

### 1) Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Start the app

```bash
streamlit run app.py
```

Streamlit will print a local URL (typically `http://localhost:8501`).

## Deploy on Streamlit Community Cloud

### 1) Push these files to GitHub

- `app.py`
- `requirements.txt`
- `runtime.txt`

### 2) Create the app in Streamlit Cloud

1. Go to https://streamlit.io/cloud and sign in with GitHub
2. Click **New app**
3. Choose:
   - Repository: `ViraatC22/Fluids-Demo`
   - Branch: `main`
   - Main file path: `app.py`
4. Click **Deploy**

### 3) If you update code later

- Push to `main`, then use **Manage app → Reboot app** to rebuild.

## Project Structure

- `app.py` — Streamlit UI + physics calculations + embedded canvas animation
- `requirements.txt` — Python dependencies for local + cloud deployment
- `runtime.txt` — Python version for Streamlit Cloud
