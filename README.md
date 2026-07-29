# Fluids Flow Lab

Fluids Flow Lab is an interactive Streamlit teaching app for exploring mass
continuity and Bernoulli's principle in an idealized pipe. It animates tracer
particles through a variable-diameter, sloped pipe; provides a local
speed/pressure probe; and lets learners record and export experimental trials.

**Status:** Complete for the documented educational scope. The physics and
primary Streamlit workflow are covered by automated tests.

## Capabilities

- Adjust flow rate, density, gravity, three pipe diameters, and endpoint heights.
- Visualize continuity-driven velocity changes with animated dots or streaks.
- Probe local diameter, speed, and ideal Bernoulli pressure by mouse or touch.
- Record configurations with calculated velocities, pressures, mass flow, and
  continuity residuals.
- Export recorded trials as UTF-8 CSV.
- Review the model equations, usage guidance, limitations, and safety notes in
  the app.

## Architecture and stack

The project intentionally remains a small single-page app:

- `app.py` contains the tested physics functions, Streamlit controls and data
  log, and a self-contained HTML canvas animation.
- Python calculates the section values shown in the data log.
- The embedded canvas applies the same continuity and Bernoulli relationships
  continuously along the visualized pipe.
- Streamlit session state stores trials for the current browser session.

Runtime: Python 3.13, Streamlit 1.52.1, and pandas 2.3.3.

## Prerequisites

- Python 3.13
- `pip`

No environment variables, credentials, database, or external API are required.

## Install and run

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

Streamlit prints the local URL, normally
[`http://localhost:8501`](http://localhost:8501).

## Verification

From the repository root, with dependencies installed:

```bash
./scripts/verify.sh
```

This command compiles the Python source and runs seven automated checks covering:

- area, velocity, mass-flow, pressure, and continuity calculations;
- constriction and elevation behavior;
- stationary flow and invalid input handling;
- initial Streamlit rendering; and
- recording a trial in the data log.

## Deploy on Streamlit Community Cloud

1. Open [Streamlit Community Cloud](https://streamlit.io/cloud) and sign in
   with GitHub.
2. Create an app from
   [`ViraatC22/Fluids-Demo`](https://github.com/ViraatC22/Fluids-Demo).
3. Select branch `main` and entry point `app.py`.
4. Deploy. `runtime.txt` requests Python 3.13 and `requirements.txt` pins the
   tested direct dependencies.

After pushing an update, reboot the app from **Manage app** if Streamlit Cloud
does not rebuild it automatically.

## Repository structure

```text
.
├── app.py                    # Physics, UI, data log, and canvas visualization
├── requirements.txt          # Pinned direct runtime dependencies
├── runtime.txt               # Streamlit Cloud Python version
├── scripts/verify.sh         # Canonical local verification command
├── tests/test_app.py         # Physics and Streamlit workflow tests
└── docs/
    ├── PROJECT_AUDIT.md
    ├── COMPLETION_PLAN.md
    └── FINAL_STATUS.md
```

## Troubleshooting

- If `streamlit` is not found, activate `.venv` and reinstall
  `requirements.txt`.
- Run commands from the repository root so the test runner can locate `app.py`.
- The trial log is session-only by design; downloading CSV is the persistence
  path.
- Very small diameters and high flow rates can produce low or negative ideal
  gauge/absolute pressure estimates. The app demonstrates the lossless equation
  and does not model cavitation or real-system operating limits.

## Model, security, and privacy limitations

This is an educational ideal-flow model. It omits viscosity, turbulence, minor
losses, pumps, cavitation, and compressibility, so it must not be used to design
pressurized hardware. It does not transmit or persist user data outside the
current Streamlit session unless the user explicitly downloads a CSV.

No license file is currently present; copyright remains with the repository
owner unless a license is added.
