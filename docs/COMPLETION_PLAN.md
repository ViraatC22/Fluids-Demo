# Completion Plan

Status values: `NOT_STARTED`, `IN_PROGRESS`, `COMPLETED`, `BLOCKED`, and
`DEFERRED_WITH_REASON`.

## Milestone 0 — Repository and environment recovery

### FD-001 — Reconcile repository state

- **Reason:** Preserve existing work and confirm the correct upstream baseline.
- **Files/modules:** Git metadata and tracked files.
- **Dependencies:** Existing `origin` access.
- **Acceptance criteria:** Clean `main`, valid remote, fetched remote metadata,
  and no divergence before edits.
- **Verification:** `git status --short --branch`, `git remote -v`,
  `git fetch --prune origin`.
- **Status:** `COMPLETED`
- **Commit:** Baseline `8658e9a` (no change required).

### FD-002 — Make dependency installation deterministic

- **Reason:** Unbounded direct dependencies make future deployments
  non-reproducible.
- **Files/modules:** `requirements.txt`, `app.py`, `.gitignore`.
- **Dependencies:** FD-001.
- **Acceptance criteria:** Tested direct versions are pinned, unused direct
  imports are removed, and common generated test output is ignored.
- **Verification:** Import during `./scripts/verify.sh`.
- **Status:** `COMPLETED`
- **Commit:** `33d6b04`

## Milestone 1 — Calculation-path restoration

### FD-101 — Consolidate and validate physics calculations

- **Reason:** Duplicate formulas could drift from the existing reusable helper.
- **Files/modules:** `app.py`.
- **Dependencies:** FD-002.
- **Acceptance criteria:** Logged middle/right values use the helper; invalid
  physical and non-finite inputs fail explicitly; valid zero flow remains
  supported.
- **Verification:** Physics tests in `tests/test_app.py`.
- **Status:** `COMPLETED`
- **Commit:** `33d6b04`

## Milestone 2 — Tests and quality enforcement

### FD-201 — Add meaningful regression coverage

- **Reason:** The initial project had no automated quality gate.
- **Files/modules:** `tests/test_app.py`, `scripts/verify.sh`.
- **Dependencies:** FD-101.
- **Acceptance criteria:** Tests cover area, continuity, constriction pressure,
  elevation pressure, zero flow, invalid inputs, Streamlit rendering, and trial
  logging.
- **Verification:** `./scripts/verify.sh`.
- **Status:** `COMPLETED`
- **Commit:** `33d6b04`

## Milestone 3 — Documentation and final verification

### FD-301 — Synchronize operational documentation

- **Reason:** Users and maintainers need accurate setup, verification,
  architecture, deployment, and limitation guidance.
- **Files/modules:** `README.md`, `docs/PROJECT_AUDIT.md`,
  `docs/COMPLETION_PLAN.md`, `docs/FINAL_STATUS.md`.
- **Dependencies:** FD-201.
- **Acceptance criteria:** Documentation matches verified behavior and commands.
- **Verification:** Manual command review plus `git diff --check`.
- **Status:** `COMPLETED`
- **Commit:** Documentation finalization commit containing this plan.

### FD-302 — Validate and publish final project state

- **Reason:** Completion requires real startup, browser smoke testing, full diff
  review, a secret scan, clean Git, and upstream synchronization.
- **Files/modules:** Entire repository.
- **Dependencies:** FD-301.
- **Acceptance criteria:** Verification passes, local app responds, browser
  principal flow is exercised through Streamlit's test harness, scans find no
  secret material, and `main` equals `origin/main`. Interactive browser control
  is attempted and any environment limitation is recorded honestly.
- **Verification:** `./scripts/verify.sh`, local Streamlit health check, browser
  interaction, `git diff --check`, secret scan, `git status --short --branch`.
- **Status:** `COMPLETED`
- **Commit:** Documentation finalization commit containing this plan.

## Deferred optional scope

- A keyboard-operable continuous pipe probe is `DEFERRED_WITH_REASON`: the data
  table and text explanation expose the core result, while a fully accessible
  canvas interaction would require a new interface design.
- Viscous/turbulent loss models and persistent server-side trial storage are
  `DEFERRED_WITH_REASON`: neither is supported by the original product evidence.
