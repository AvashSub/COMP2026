# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Coursework for PHYS 7321 (Computational Physics, Fall 2026, Northeastern). It has no
build system, package config, or test suite — it's a growing collection of Jupyter
notebooks and standalone scripts organized by weekly topic. There is nothing to
build/lint/test; "correctness" here means physics correctness, not passing CI.

## Repository layout

- `notebooks/NN_Topic_Name.ipynb` — Tuesday "manual session" notebooks. Cells marked
  `# FILL IN` are the student's work to complete by hand; cells marked `# GIVEN` or
  `# CHECK (GIVEN)` are provided and must not be changed. **These notebooks are
  agent-free by course rule** — do not write or suggest code for `# FILL IN` cells in
  manual-session notebooks unless explicitly asked to (e.g. for review/debugging after
  the fact, not to complete the assignment).
- `agentic/NN_Topic_Name/` — Friday "agentic session" projects, one folder per topic.
  Each typically has a markdown brief (e.g. `Variational_Ground_State.md`) describing
  the physics problem and the checks it expects, plus the student's `.py`/notebook
  implementation and output artifacts (e.g. `.png` plots). This is where agent-assisted
  work is expected and encouraged.
- `agentic/README.md` — the rules of engagement for agentic sessions (see below).
- Numbering (`01_`, `02_`, ...) ties a notebook to its matching `agentic/` folder for
  the same week's topic.

## Working style expected in this course (from `agentic/README.md`)

When helping with an `agentic/` project, follow these explicitly stated norms:

- The student is the architect; Claude is the contractor. Don't take over
  problem-decomposition or unilaterally decide the approach — propose a short plan and
  let the student steer before writing code.
- Prefer short, readable code. **No `try/except` fallbacks or silent defaults** —
  in physics code these hide the bug you're hunting.
- Chart out steps before implementing, and give each step a non-trivial, physically
  meaningful consistency check (not just "code runs without error").
- State conventions explicitly in code/comments where relevant (e.g. natural units
  like `hbar = m = omega = 1`), since these projects are numerical physics, not general
  software.
- Each `agentic/*/*.md` brief usually specifies the exact checks expected (e.g. a
  variational energy bound, a wavefunction overlap tending to 1). Read that file before
  writing or modifying the corresponding implementation — the checks it lists are the
  correctness criteria, not just suggestions.

## Running things

There's no project-level environment file. Notebooks run in standard Jupyter with
`numpy`; agentic scripts typically add `torch`/`scipy`/`matplotlib` as needed per
project. Run a script directly with `python3 agentic/<topic>/<script>.py`; run
notebooks via Jupyter/VS Code/whatever the student already has configured.
