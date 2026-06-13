# Effect of Hydrogen Enrichment on Methane-Air Combustion

MKWS (Computer Methods in Combustion) project — Warsaw University of Technology,
Faculty of Power and Aeronautical Engineering.

Laminar flame speed, ignition delay and NOx emissions of CH4/H2/air mixtures,
computed with [Cantera](https://cantera.org) using the GRI-Mech 3.0 mechanism,
for hydrogen blend fractions X_H2 in [0, 1].

## Project structure

- `src/` — Python source code (Cantera simulations)
  - `mixtures.py` — shared helpers (fuel/oxidizer setup, LHV calculation)
  - `equilibrium_study.py` — adiabatic flame temperature & equilibrium NOx/CO
  - `flame_speed_study.py` — 1D laminar flame speed (FreeFlame)
  - `ignition_delay_study.py` — 0D ignition delay (IdealGasReactor)
  - `main.py` — runs all studies
- `data/` — CSV results from each study
- `figures/` — generated plots (PNG)
- `report/` — LaTeX project report (`main.tex`, `references.bib`)

## Running the simulations

```bash
pip install -r requirements.txt
cd src
python main.py        # run everything
python main.py eq     # equilibrium study only
python main.py flame  # flame speed study only (slow, ~10-15 min)
python main.py ign    # ignition delay study only
```

Results are written to `data/` and `figures/`.
