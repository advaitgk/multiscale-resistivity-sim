# Multiscale Simulation of Electrical Resistivity in Sn-Bi Solder Joints

A from-scratch, multiscale computational pipeline that estimates the effective
electrical resistivity of near-eutectic Sn-Bi solder microstructures by
explicitly simulating the underlying physics — phase microstructure, solute
diffusion, impurity scattering, and steady-state current transport — rather than
relying on simple geometric mixing rules.

> Undergraduate research project (UGP), Dept. of Materials Science & Engineering,
> IIT Kanpur. Author: **Advait Girish Karmarkar**. Advisor: **Prof. Nilesh Badwe**.

---

## Motivation

In Sn-Bi solder joints, impurity atoms are kinetically trapped at phase
boundaries during rapid cooling, producing localized concentrations that far
exceed equilibrium solubility (e.g. up to ~21% Bi in Sn). These trapped impurities
scatter conduction electrons and create electrical "bottlenecks" that simple
parallel/series resistivity models cannot capture. This project models those
bottlenecks directly.

## Method — a four-stage pipeline (+ dimensionality correction)

| Stage | What it does | Key technique |
|-------|--------------|---------------|
| 1. Microstructure | Grow a two-phase Sn/Bi morphology from a random melt | Ising/Metropolis **Kinetic Monte Carlo** (Numba JIT) |
| 2. Diffusion | Redistribute trapped solute toward grain cores | **Fick's 2nd law**, FTCS finite differences |
| 3. Scattering | Map local impurity excess to local resistivity | Linearized, Nordheim-type penalty `ρ = ρ_base + k·ΔC` |
| 4. Transport | Solve for the steady-state voltage field and effective resistivity | **Sparse finite-difference resistor network**, Conjugate Gradient solver (10⁶ nodes) |
| 5. 2D → 3D | Lift the planar result to a 3D estimate | **Bakker (1997) effective-medium theory** |

The full implementation is a single, self-contained script:
[`implementation/final_implementation.py`](implementation/final_implementation.py).

## Result

On a 1000×1000 grid (10⁶ nodes), the pipeline produces a planar (2D) effective
resistivity of ~72 µΩ·cm, which the Bakker 2D→3D correction lifts to a 3D
estimate of **~53 µΩ·cm** — within the experimentally reported range of
35–60 µΩ·cm for near-eutectic Sn-Bi.

![pipeline output](report/) <!-- output figures are shown in report/final_presentation.pdf -->

## Repository structure

```
implementation/   Authoritative end-to-end pipeline (final_implementation.py)
development/       Development notebook documenting the project's evolution
report/            Final technical report and presentation (PDF)
REFERENCES.md      Literature the project builds on (PDFs not redistributed)
requirements.txt   Python dependencies
```

## Getting started

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python implementation/final_implementation.py
```

The script prints the 2D and 3D-corrected resistivity and displays a four-panel
figure (phase map, impurity distribution, resistivity map, voltage field).
Note: the 1000×1000 solve uses several GB of RAM; reduce `GRID_SIZE` to run on a
smaller machine.

## Limitations & scope (honest notes)

This is a phenomenological model intended to capture the *mechanism* of
boundary-driven resistivity, not a calibration-free first-principles prediction:

- The microstructure is a 2D slice; the 3D value comes from an effective-medium
  extrapolation (Bakker EMT) rather than a full 3D solve.
- Several parameters (interface penalty, scattering coefficients) are calibrated
  against literature resistivity values.
- Results shown are from a single stochastic realization.

See the [report](report/final_report.pdf) for the full methodology, and the
[development notebook](development/project_notebook.ipynb) for how the model
evolved. Planned improvements (seeded ensembles + uncertainty bands, a direct 3D
solver, and validation against temperature-dependent measurements) are natural
next steps.

## References

See [REFERENCES.md](REFERENCES.md). The 2D→3D correction follows Bakker (1997),
*Int. J. Heat and Mass Transfer* 40(15):3503–3511.

## License

[MIT](LICENSE) © 2026 Advait Girish Karmarkar.
