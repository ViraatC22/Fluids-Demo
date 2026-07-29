# Project Audit

Audit date: 2026-07-29

## 1. Project purpose

Fluids Flow Lab is a Streamlit educational simulation for exploring continuity
and Bernoulli's principle in a variable-diameter, sloped pipe. Repository
evidence for that purpose is consistent across the README, app copy, controls,
equations, animation, data-log fields, and five-commit Git history.

## 2. Existing architecture

- A single Python entry point (`app.py`) hosts the Streamlit UI and pure physics
  helpers.
- A self-contained HTML/JavaScript canvas, embedded by Streamlit, renders the
  pipe, particle motion, and local hover/touch probe.
- pandas stores trial rows in Streamlit session state and exports CSV.
- `requirements.txt` and `runtime.txt` define the Streamlit Cloud runtime.
- `origin` points to the private or public status-controlled upstream repository
  `ViraatC22/Fluids-Demo`; local `main` began the audit clean and synchronized at
  `8658e9a`.

## 3. Current functionality

The app renders without Streamlit exceptions. Eight sliders control flow,
density, gravity, geometry, and elevation; three toggles control animation.
Users can inspect the simulation, read the explanation, record trials, view
calculated results, and export CSV. The lossless continuity and Bernoulli
calculations produce physically consistent section results.

## 4. Broken functionality found

No user-visible crash was reproduced. The highest-risk defect was structural:
the UI reimplemented its section calculations instead of calling the existing
physics helper. That allowed the tested helper and displayed results to diverge
silently. The helper also accepted non-finite and nonphysical inputs.

## 5. Missing functionality

No critical feature supported by repository evidence was absent. Persistent
server-side storage, authentication, real-fluid losses, and hardware-design
features are outside the demonstrated educational scope.

## 6. Build and runtime problems

The app imported and rendered on the available Python 3.14 environment, while
deployment requests Python 3.13. The original dependency file used unbounded
versions and declared NumPy even though the code did not use it directly. There
was no canonical verification command.

## 7. Dependency problems

Direct runtime dependencies are now pinned to the versions exercised during
recovery: Streamlit 1.52.1 and pandas 2.3.3. NumPy remains an indirect pandas
dependency but is no longer declared as if the application imported it.

## 8. Security concerns

No credentials, private keys, environment files, or secret-like values were
found in tracked project content. The application takes only bounded numeric
inputs through Streamlit and does not make network, filesystem-write, database,
authentication, or shell calls. Embedded HTML contains only project-controlled
numeric and Boolean values. No in-scope security defect was identified.

## 9. Testing gaps

The initial repository had no automated tests. It lacked regression coverage
for continuity, Bernoulli pressure, elevation, stationary flow, invalid inputs,
initial UI rendering, and trial logging.

## 10. Documentation gaps

The original README documented only basic setup and deployment. It did not state
project status, architecture, verification, prerequisites, runtime behavior,
troubleshooting, privacy, limitations, or licensing status.

## 11. Deployment gaps

Streamlit Cloud configuration was present through `runtime.txt`,
`requirements.txt`, and README instructions. Automated deployment was not
evidenced or necessary. A live production deployment was not available for
inspection during this local recovery.

## 12. Accessibility and usability gaps

The local probe supports pointer events for mouse and touch, controls use native
Streamlit widgets, and the explanatory content provides text alternatives to
the animation. The canvas itself is not keyboard-operable and has no equivalent
continuous-position screen-reader probe. The data log and explanation preserve
the core numerical and conceptual information; a fully accessible canvas
replacement would be a larger optional enhancement.

## 13. Completion definition

The evidenced scope is complete when:

1. The UI uses one validated calculation path for logged section values.
2. Core equations and the record-trial workflow have meaningful automated tests.
3. One documented command runs syntax and test checks.
4. Direct dependencies are reproducible and unused imports are removed.
5. The README accurately documents setup, operation, verification, deployment,
   limitations, and repository status.
6. The app starts successfully and the primary UI workflow is smoke-tested with
   Streamlit's supported test harness.
7. Git is clean, reviewed, secret-scanned, committed, and synchronized.

## 14. Prioritized implementation plan

1. Consolidate section calculations on the pure helper and validate its domain.
2. Add regression tests for equations, error behavior, rendering, and logging.
3. Add a canonical verification script and pin direct dependencies.
4. Synchronize operational documentation.
5. Run a real-server health check, attempt an interactive browser inspection,
   inspect diffs, and publish the verified milestone.

## 15. Known blockers and assumptions

There are no local completion blockers. Deployment validation is limited to
configuration and local runtime behavior because no deployed app URL was
recorded in the repository. The model is intentionally ideal and lossless;
expanding it into a real-fluid engineering tool would be a new product decision.
