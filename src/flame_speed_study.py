"""
Laminar flame speed study for CH4/H2/air mixtures using ct.FreeFlame (1D).

Computes:
  - S_L vs equivalence ratio phi, for several H2 blend fractions X_H2
    (also gives S_L vs X_H2 at phi=1 as a cross-section)
  - S_L vs pressure, for X_H2 = 0.0, 0.5, 1.0
  - flame structure (T and major species profiles) for one reference case

Each flame solve takes ~10-25 s, so results are cached incrementally to CSV
files in data/ and the script can be re-run safely: already-computed
(x_h2, phi, P) combinations are skipped.

Outputs:
  - figures/05_SL_vs_phi.png
  - figures/06_SL_vs_xH2.png
  - figures/07_SL_vs_pressure.png
  - figures/08_flame_structure.png
  - data/flame_speed_phi_scan.csv
  - data/flame_speed_pressure_scan.csv
"""

from pathlib import Path

import cantera as ct
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mixtures import make_gas, set_mixture

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures"
DATA_DIR = ROOT / "data"

T0 = 300.0  # K
P0 = ct.one_atm
WIDTH = 0.03  # m

X_H2_LIST = [0.0, 0.25, 0.5, 0.75, 1.0]
PHI_LIST = [0.7, 0.85, 1.0, 1.15, 1.3]
P_LIST_BAR = [1.0, 3.0, 6.0]
X_H2_PRESSURE = [0.0, 0.5, 1.0]

PHI_SCAN_CSV = DATA_DIR / "flame_speed_phi_scan.csv"
PRESSURE_SCAN_CSV = DATA_DIR / "flame_speed_pressure_scan.csv"


def solve_flame(phi: float, x_h2: float, T: float = T0, P: float = P0, width: float = WIDTH):
    """Solve a freely-propagating premixed flame, return (flame, Su [m/s])."""
    gas = make_gas()
    set_mixture(gas, phi, x_h2, T, P)
    flame = ct.FreeFlame(gas, width=width)
    flame.set_refine_criteria(ratio=3, slope=0.1, curve=0.2)
    flame.solve(loglevel=0, auto=True)
    return flame, flame.velocity[0]


def _load_cache(path: Path, key_cols) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in key_cols + ["Su"]})


def run_phi_scan() -> pd.DataFrame:
    key_cols = ["x_h2", "phi"]
    df = _load_cache(PHI_SCAN_CSV, key_cols)
    done = set(zip(df.x_h2.round(6), df.phi.round(6)))

    for x_h2 in X_H2_LIST:
        for phi in PHI_LIST:
            key = (round(x_h2, 6), round(phi, 6))
            if key in done:
                continue
            print(f"  [phi scan] X_H2={x_h2:.2f}, phi={phi:.2f} ...", flush=True)
            try:
                _, Su = solve_flame(phi, x_h2)
            except ct.CanteraError as exc:
                print(f"    FAILED: {exc}")
                Su = np.nan
            df = pd.concat([df, pd.DataFrame([{"x_h2": x_h2, "phi": phi, "Su": Su}])],
                            ignore_index=True)
            df.to_csv(PHI_SCAN_CSV, index=False)
            done.add(key)
    return df


def run_pressure_scan() -> pd.DataFrame:
    key_cols = ["x_h2", "P_bar"]
    df = _load_cache(PRESSURE_SCAN_CSV, key_cols)
    done = set(zip(df.x_h2.round(6), df.P_bar.round(6)))

    for x_h2 in X_H2_PRESSURE:
        for P_bar in P_LIST_BAR:
            key = (round(x_h2, 6), round(P_bar, 6))
            if key in done:
                continue
            print(f"  [pressure scan] X_H2={x_h2:.2f}, P={P_bar:.1f} bar ...", flush=True)
            try:
                _, Su = solve_flame(1.0, x_h2, P=P_bar * 1e5)
            except ct.CanteraError as exc:
                print(f"    FAILED: {exc}")
                Su = np.nan
            df = pd.concat([df, pd.DataFrame([{"x_h2": x_h2, "P_bar": P_bar, "Su": Su}])],
                            ignore_index=True)
            df.to_csv(PRESSURE_SCAN_CSV, index=False)
            done.add(key)
    return df


def plot_SL_vs_phi(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for x_h2 in X_H2_LIST:
        d = df[df.x_h2 == x_h2].sort_values("phi")
        ax.plot(d.phi, d.Su * 100, "o-", linewidth=2, label=f"X_H2 = {x_h2:.2f}")
    ax.axvline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Equivalence ratio, phi [-]")
    ax.set_ylabel("Laminar flame speed, S_L [cm/s]")
    ax.set_title("Laminar flame speed vs equivalence ratio\nCH4/H2/air blends, T0=300 K, p=1 atm")
    ax.legend(title="Fuel blend", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_SL_vs_phi.png", dpi=150)
    plt.close(fig)


def plot_SL_vs_xh2(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    d = df[np.isclose(df.phi, 1.0)].sort_values("x_h2")
    ax.plot(d.x_h2, d.Su * 100, "o-", linewidth=2, color="darkred")
    ax.set_xlabel("H2 fraction in fuel, X_H2 [-]")
    ax.set_ylabel("Laminar flame speed, S_L [cm/s]")
    ax.set_title("Laminar flame speed vs H2 blend fraction\nstoichiometric CH4/H2/air, T0=300 K, p=1 atm")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_SL_vs_xH2.png", dpi=150)
    plt.close(fig)


def plot_SL_vs_pressure(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for x_h2 in X_H2_PRESSURE:
        d = df[df.x_h2 == x_h2].sort_values("P_bar")
        ax.plot(d.P_bar, d.Su * 100, "o-", linewidth=2, label=f"X_H2 = {x_h2:.2f}")
    ax.set_xlabel("Pressure [bar]")
    ax.set_ylabel("Laminar flame speed, S_L [cm/s]")
    ax.set_title("Laminar flame speed vs pressure\nstoichiometric CH4/H2/air, T0=300 K")
    ax.legend(title="Fuel blend", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_SL_vs_pressure.png", dpi=150)
    plt.close(fig)


def plot_flame_structure(x_h2: float = 0.5, phi: float = 1.0):
    print(f"  [structure] X_H2={x_h2:.2f}, phi={phi:.2f} ...", flush=True)
    flame, Su = solve_flame(phi, x_h2)
    z = flame.grid * 1000  # mm

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    axes[0].plot(z, flame.T, "k-", linewidth=2)
    axes[0].set_ylabel("Temperature [K]")
    axes[0].set_title(f"Flame structure: X_H2={x_h2:.2f}, phi={phi:.2f}, "
                       f"S_L={Su*100:.1f} cm/s, T_ad={flame.T[-1]:.0f} K")
    axes[0].grid(alpha=0.3)

    for sp in ["CH4", "H2", "O2", "CO2", "H2O", "OH"]:
        axes[1].plot(z, flame.X[flame.gas.species_index(sp)], linewidth=2, label=sp)
    axes[1].set_xlabel("Position [mm]")
    axes[1].set_ylabel("Mole fraction [-]")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "08_flame_structure.png", dpi=150)
    plt.close(fig)


def run():
    FIG_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)

    print("Flame speed study: phi scan...")
    df_phi = run_phi_scan()
    plot_SL_vs_phi(df_phi)
    plot_SL_vs_xh2(df_phi)

    print("Flame speed study: pressure scan...")
    df_p = run_pressure_scan()
    plot_SL_vs_pressure(df_p)

    print("Flame speed study: flame structure...")
    plot_flame_structure(x_h2=0.5, phi=1.0)

    print("\nDone. Figures saved to:", FIG_DIR)


if __name__ == "__main__":
    run()
