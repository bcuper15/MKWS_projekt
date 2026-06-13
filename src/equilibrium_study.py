"""
Equilibrium combustion study for CH4/H2/air mixtures.

Computes adiabatic flame temperature (HP equilibrium) and equilibrium
product composition (CO2, H2O, CO, NO, NO2, OH) as a function of:

  - equivalence ratio phi, for several H2 blend fractions X_H2
  - H2 blend fraction X_H2, at fixed phi

Outputs:
  - figures/01_Tad_vs_phi.png
  - figures/02_equilibrium_NOx_vs_phi.png
  - figures/03_equilibrium_CO_vs_phi.png
  - figures/04_NOx_CO_vs_xH2.png
  - data/equilibrium_results.csv
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

X_H2_LIST = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
PHI_ARRAY = np.linspace(0.5, 1.6, 45)

TRACE_SPECIES = ["CO2", "H2O", "CO", "NO", "NO2", "OH", "H2", "O2"]


def equilibrium_state(phi: float, x_h2: float, T: float = T0, P: float = P0):
    """Return the gas object after HP-equilibration at the given state."""
    gas = make_gas()
    set_mixture(gas, phi, x_h2, T, P)
    gas.equilibrate("HP")
    return gas


def scan_phi(x_h2: float, phi_array: np.ndarray = PHI_ARRAY) -> pd.DataFrame:
    """Equilibrium results vs phi for a fixed H2 blend fraction."""
    rows = []
    for phi in phi_array:
        gas = equilibrium_state(phi, x_h2)
        row = {"phi": phi, "x_h2": x_h2, "T_ad": gas.T}
        for sp in TRACE_SPECIES:
            row[sp] = gas[sp].X[0]
        rows.append(row)
    return pd.DataFrame(rows)


def scan_x_h2(phi: float, x_h2_array=X_H2_LIST) -> pd.DataFrame:
    """Equilibrium results vs X_H2 for a fixed equivalence ratio."""
    rows = []
    for x_h2 in x_h2_array:
        gas = equilibrium_state(phi, x_h2)
        row = {"phi": phi, "x_h2": x_h2, "T_ad": gas.T}
        for sp in TRACE_SPECIES:
            row[sp] = gas[sp].X[0]
        rows.append(row)
    return pd.DataFrame(rows)


def plot_Tad_vs_phi(df_all: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for x_h2 in X_H2_LIST:
        d = df_all[df_all.x_h2 == x_h2]
        ax.plot(d.phi, d.T_ad, linewidth=2, label=f"X_H2 = {x_h2:.1f}")
    ax.axvline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Equivalence ratio, phi [-]")
    ax.set_ylabel("Adiabatic flame temperature, T_ad [K]")
    ax.set_title("Adiabatic flame temperature vs equivalence ratio\nCH4/H2/air blends, T0=300 K, p=1 atm")
    ax.legend(title="Fuel blend", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_Tad_vs_phi.png", dpi=150)
    plt.close(fig)


def plot_NOx_vs_phi(df_all: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for x_h2 in X_H2_LIST:
        d = df_all[df_all.x_h2 == x_h2]
        nox_ppm = (d.NO + d.NO2) * 1e6
        ax.semilogy(d.phi, nox_ppm, linewidth=2, label=f"X_H2 = {x_h2:.1f}")
    ax.axvline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Equivalence ratio, phi [-]")
    ax.set_ylabel("Equilibrium NOx (NO+NO2) [ppm]")
    ax.set_title("Equilibrium NOx vs equivalence ratio\nCH4/H2/air blends, T0=300 K, p=1 atm")
    ax.legend(title="Fuel blend", fontsize=9)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_equilibrium_NOx_vs_phi.png", dpi=150)
    plt.close(fig)


def plot_CO_vs_phi(df_all: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for x_h2 in X_H2_LIST:
        d = df_all[df_all.x_h2 == x_h2]
        co_ppm = d.CO * 1e6
        ax.semilogy(d.phi, co_ppm, linewidth=2, label=f"X_H2 = {x_h2:.1f}")
    ax.axvline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Equivalence ratio, phi [-]")
    ax.set_ylabel("Equilibrium CO [ppm]")
    ax.set_title("Equilibrium CO vs equivalence ratio\nCH4/H2/air blends, T0=300 K, p=1 atm")
    ax.legend(title="Fuel blend", fontsize=9)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_equilibrium_CO_vs_phi.png", dpi=150)
    plt.close(fig)


def plot_NOx_CO_vs_xh2(df_lean: pd.DataFrame, df_stoich: pd.DataFrame, df_rich: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for df, label, style in [
        (df_lean, "phi = 0.8", "o-"),
        (df_stoich, "phi = 1.0", "s-"),
        (df_rich, "phi = 1.2", "^-"),
    ]:
        axes[0].semilogy(df.x_h2, (df.NO + df.NO2) * 1e6, style, linewidth=2, label=label)
        axes[1].semilogy(df.x_h2, df.CO * 1e6 + 1e-6, style, linewidth=2, label=label)

    axes[0].set_xlabel("H2 fraction in fuel, X_H2 [-]")
    axes[0].set_ylabel("Equilibrium NOx (NO+NO2) [ppm]")
    axes[0].set_title("NOx vs H2 blend fraction")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3, which="both")

    axes[1].set_xlabel("H2 fraction in fuel, X_H2 [-]")
    axes[1].set_ylabel("Equilibrium CO [ppm]")
    axes[1].set_title("CO vs H2 blend fraction")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3, which="both")

    fig.suptitle("Effect of H2 enrichment on equilibrium NOx and CO (T0=300 K, p=1 atm)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_NOx_CO_vs_xH2.png", dpi=150)
    plt.close(fig)


def run():
    FIG_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)

    print("Equilibrium study: scanning phi for each H2 blend fraction...")
    df_all = pd.concat([scan_phi(x_h2) for x_h2 in X_H2_LIST], ignore_index=True)
    df_all.to_csv(DATA_DIR / "equilibrium_results.csv", index=False)

    plot_Tad_vs_phi(df_all)
    plot_NOx_vs_phi(df_all)
    plot_CO_vs_phi(df_all)

    print("Equilibrium study: scanning X_H2 at fixed phi...")
    x_h2_fine = np.linspace(0.0, 1.0, 11)
    df_lean = scan_x_h2(0.8, x_h2_fine)
    df_stoich = scan_x_h2(1.0, x_h2_fine)
    df_rich = scan_x_h2(1.2, x_h2_fine)
    pd.concat([df_lean, df_stoich, df_rich], ignore_index=True).to_csv(
        DATA_DIR / "equilibrium_vs_xh2.csv", index=False
    )

    plot_NOx_CO_vs_xh2(df_lean, df_stoich, df_rich)

    # Summary printout
    idx_max = df_all.groupby("x_h2")["T_ad"].idxmax()
    print("\nMax T_ad per blend (and the phi at which it occurs):")
    for x_h2, i in idx_max.items():
        row = df_all.loc[i]
        print(f"  X_H2={x_h2:.1f}: T_ad_max = {row.T_ad:.1f} K at phi = {row.phi:.2f}")

    print("\nDone. Figures saved to:", FIG_DIR)


if __name__ == "__main__":
    run()
